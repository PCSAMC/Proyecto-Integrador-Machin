# Grupo 4 — Predicción de Precios Inmobiliarios (Ames Housing)
## Machine Learning 2026-I | SIS-351

**Integrantes:** Jean Marco · Sergio Arias · Marvin Mollo · Jaime Huaycho · Sergio Mendoza  
**Metodología:** CRISP-DM  
**Dataset:** Ames Housing — OpenML ID: 41211

---

## Estructura del Proyecto

```
grupo4_ames_housing/
│
├── data/                        # Datos originales y procesados
│   ├── raw/                     # Dataset crudo descargado de OpenML
│   ├── processed/               # Datos limpios y particionados
│   └── metadata.json            # Trazabilidad: fuente, versión, fecha
│
├── notebooks/                   # Jupyter Notebooks por fase CRISP-DM
│   ├── 01_EDA.ipynb             # FASE 2 — EDA y Data Understanding
│   ├── 02_Preprocessing.ipynb   # FASE 2 — Data Preparation
│   ├── 03_ML_Models.ipynb       # FASE 2 — Modelos ML + AutoML
│   └── 04_Clustering.ipynb      # FASE 2 — Componente No Supervisado
│
├── src/                         # Scripts Python reutilizables
│   ├── data_loader.py           # Descarga y carga del dataset
│   ├── preprocessing.py         # Pipeline de preprocesamiento
│   ├── models.py                # Entrenamiento y evaluación de modelos
│   ├── clustering.py            # K-Means y visualización de clusters
│   └── utils.py                 # Funciones auxiliares y métricas
│
├── models/                      # Artefactos serializados
│   └── (generados al ejecutar)
│
├── reports/                     # Informes y figuras exportadas
│   └── figures/
│
├── app/                         # Demo (Fase 4 — MLOps)
│
├── config/
│   └── params.yaml              # Hiperparámetros y configuración
│
├── requirements.txt             # Dependencias fijadas
└── README.md                    # Este archivo
```

---

## Instalación y Ejecución

### 1. Clonar / descargar el proyecto
```bash
cd grupo4_ames_housing
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar los notebooks en orden
Abrir Jupyter y ejecutar en secuencia:
```
notebooks/01_EDA.ipynb
notebooks/02_Preprocessing.ipynb
notebooks/03_ML_Models.ipynb
notebooks/04_Clustering.ipynb
```

> **Nota Colab:** Los notebooks incluyen la celda `!pip install -r requirements.txt` al inicio. Solo ejecútalos en orden desde Colab.

---

## Fases del Proyecto (CRISP-DM)

| Fase | Notebook | Estado |
|------|----------|--------|
| Business Understanding | — (ver propuesta) | ✅ Completo |
| Data Understanding (EDA) | `01_EDA.ipynb` | ✅ Completo |
| Data Preparation | `02_Preprocessing.ipynb` | ✅ Completo |
| Modeling — ML Clásico + AutoML | `03_ML_Models.ipynb` | ✅ Completo |
| Modeling — Clustering | `04_Clustering.ipynb` | ✅ Completo |
| Evaluation | Integrado en notebooks 03 y 04 | ✅ Completo |
| Deployment (MLOps + RAG) | `app/` | 🔜 Fase 4 |

---

## Dataset

- **Fuente:** OpenML — https://www.openml.org/search?type=data&status=active&id=41211
- **Autor:** Dean De Cock (2011)
- **Registros:** 2,930 viviendas
- **Variables:** 81 (80 predictoras + Sale_Price)
- **Variable objetivo:** `Sale_Price` (USD, continua)

---

## Reproducibilidad

- Semilla fija: `RANDOM_STATE = 42` en todos los scripts
- Partición: 70% train / 15% val / 15% test (estratificada por precio)
- Pipeline sklearn ajustado solo en train; aplicado en val/test
