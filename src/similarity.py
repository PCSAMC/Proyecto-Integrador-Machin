"""
src/similarity.py
=================
Búsqueda de propiedades similares usando embeddings del MLP/Autoencoder.

CRISP-DM: Deployment (RAG Tabular)
- Dado un índice de propiedad, retorna las K más similares por coseno
- Soporta FAISS (rápido, producción) y NearestNeighbors (fallback)
- Precisión@K: qué % de vecinos caen en el mismo segmento de precio
- Visualización de propiedades similares con sus características clave
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ─── Clase principal ──────────────────────────────────────────────────────────

class PropertySimilarity:
    """
    Búsqueda de propiedades similares sobre el espacio latente 128D del MLP.

    Uso
    ---
    sim = PropertySimilarity()
    sim.fit('models/mlp_embeddings.npy', 'models/mlp_embeddings_labels.npy')
    result = sim.search(query_idx=42, k=5)
    p_at_k = sim.precision_at_k(k=5)
    """

    def __init__(self, metric: str = "cosine"):
        self.metric     = metric
        self.embeddings = None
        self.prices_usd = None
        self.segments   = None
        self._nn        = None

    # ── Carga ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        embeddings_path: str = "models/mlp_embeddings.npy",
        labels_path:     str = "models/mlp_embeddings_labels.npy",
        raw_df:          pd.DataFrame = None,
        use_faiss:       bool = True,
    ):
        """
        Carga embeddings y precios. Construye el índice de búsqueda.

        use_faiss: usa FAISS si está disponible (más rápido en producción).
                   Hace fallback a NearestNeighbors si faiss no está instalado.
        raw_df: DataFrame con datos crudos (opcional) para mostrar features clave.
        """
        self.embeddings = np.load(embeddings_path).astype(np.float32)
        self.prices_usd = np.load(labels_path).astype(np.float32)
        self.raw_df     = raw_df

        emb_norm = normalize(self.embeddings, norm="l2")

        self._use_faiss = use_faiss and FAISS_AVAILABLE
        if self._use_faiss:
            # FAISS IndexFlatIP = producto interno sobre vectores L2-normalizados
            # equivale a similitud coseno
            d = emb_norm.shape[1]
            self._faiss_index = faiss.IndexFlatIP(d)
            self._faiss_index.add(emb_norm)
            self._emb_norm = emb_norm
            backend = "FAISS (IndexFlatIP)"
        else:
            self._nn = NearestNeighbors(
                n_neighbors=20,
                metric=self.metric,
                algorithm="brute",
                n_jobs=-1,
            )
            self._nn.fit(emb_norm)
            self._emb_norm = emb_norm
            backend = "NearestNeighbors (sklearn)"

        self.segments = self._assign_segments(self.prices_usd)
        print(f"[Similarity] {len(self.embeddings)} propiedades indexadas | backend={backend} | dim={emb_norm.shape[1]}")
        return self

    # ── Búsqueda ──────────────────────────────────────────────────────────────

    def _query(self, query_vec: np.ndarray, k: int):
        """Búsqueda interna usando FAISS o NearestNeighbors."""
        if self._use_faiss:
            scores, indices = self._faiss_index.search(query_vec, k)
            return scores[0], indices[0]          # scores = similitud coseno (IP)
        else:
            distances, indices = self._nn.kneighbors(query_vec, n_neighbors=k)
            return 1 - distances[0], indices[0]   # convertir distancia coseno a similitud

    def search(self, query_idx: int, k: int = 5) -> pd.DataFrame:
        """
        Retorna las K propiedades más similares a query_idx.
        Excluye la propiedad query en los resultados.
        """
        query_vec = self._emb_norm[query_idx].reshape(1, -1)

        sims, indices = self._query(query_vec, k + 1)

        # Excluir la propiedad misma
        mask    = indices != query_idx
        indices = indices[mask][:k]
        sims    = sims[mask][:k]

        result = pd.DataFrame({
            "idx":          indices,
            "similarity":   np.round(sims, 4),
            "price_usd":    self.prices_usd[indices].astype(int),
            "segment":      self.segments[indices],
        })

        # Agregar features clave si hay raw_df
        if self.raw_df is not None:
            key_cols = [c for c in ["Gr_Liv_Area", "Overall_Qual", "Year_Built",
                                     "Neighborhood", "Total_Bsmt_SF", "Garage_Cars"]
                        if c in self.raw_df.columns]
            extra = self.raw_df.iloc[indices][key_cols].reset_index(drop=True)
            result = pd.concat([result.reset_index(drop=True), extra], axis=1)

        return result

    def search_by_vector(self, embedding: np.ndarray, k: int = 5) -> pd.DataFrame:
        """
        Búsqueda dado un vector embedding directamente (para propiedades nuevas).
        """
        query_vec = normalize(embedding.reshape(1, -1), norm="l2")
        sims, indices = self._query(query_vec, k)

        result = pd.DataFrame({
            "idx":        indices,
            "similarity": np.round(sims, 4),
            "price_usd":  self.prices_usd[indices].astype(int),
            "segment":    self.segments[indices],
        })
        return result

    # ── Precisión@K ───────────────────────────────────────────────────────────

    def precision_at_k(self, k: int = 5, n_queries: int = 100, seed: int = 42) -> dict:
        """
        Evalúa la calidad de la recuperación.

        Definición de relevancia: el vecino está en el mismo segmento de precio
        que la query (< $120k / $120k-$200k / $200k-$300k / > $300k).

        Retorna dict con P@K global y por segmento.
        """
        rng     = np.random.default_rng(seed)
        queries = rng.choice(len(self.embeddings), size=n_queries, replace=False)

        precisions        = []
        precision_by_seg  = {s: [] for s in ["<$120k", "$120k-$200k", "$200k-$300k", ">$300k"]}

        for idx in queries:
            result       = self.search(idx, k=k)
            query_seg    = self.segments[idx]
            relevant     = (result["segment"] == query_seg).sum()
            p_at_k       = relevant / k
            precisions.append(p_at_k)
            if query_seg in precision_by_seg:
                precision_by_seg[query_seg].append(p_at_k)

        global_p = float(np.mean(precisions))

        seg_p = {
            seg: float(np.mean(vals)) if vals else 0.0
            for seg, vals in precision_by_seg.items()
        }

        print(f"\n[Precisión@{k}] Global: {global_p:.4f} ({global_p*100:.1f}%)")
        for seg, val in seg_p.items():
            print(f"  {seg:20s}: {val:.4f} ({val*100:.1f}%)")

        return {"global": global_p, "by_segment": seg_p, "k": k, "n_queries": n_queries}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _assign_segments(prices: np.ndarray) -> np.ndarray:
        segments = np.empty(len(prices), dtype=object)
        segments[prices <  120_000] = "<$120k"
        segments[(prices >= 120_000) & (prices < 200_000)] = "$120k-$200k"
        segments[(prices >= 200_000) & (prices < 300_000)] = "$200k-$300k"
        segments[prices >= 300_000] = ">$300k"
        return segments

    def summary(self) -> None:
        """Imprime un resumen del índice de similitud."""
        segs, counts = np.unique(self.segments, return_counts=True)
        print(f"\n[Similarity] Resumen del índice:")
        print(f"  Total propiedades : {len(self.embeddings)}")
        print(f"  Dimensión embedding: {self.embeddings.shape[1]}")
        print(f"  Métrica           : {self.metric}")
        print(f"  Distribución por segmento:")
        for s, c in zip(segs, counts):
            print(f"    {s:20s}: {c:4d} ({c/len(self.embeddings)*100:.1f}%)")
