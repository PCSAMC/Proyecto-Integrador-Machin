# ============================================================
#  Dockerfile — Grupo 4 Ames Housing
#  Empaqueta TODO el proyecto: el docente solo necesita Docker.
#
#  Por defecto usa requirements.txt (Linux x86/amd64).
#  En macOS Apple Silicon cambia REQS en docker-compose.yml.
# ============================================================
FROM python:3.11-slim

# Dependencias del sistema (xgboost/lightgbm requieren libgomp; faiss requiere libstdc++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# El archivo de dependencias se selecciona via build arg.
# Valor por defecto: requirements.txt (Linux/x86).
# Para macOS: REQS=requirements-macos.txt (ver docker-compose.yml).
ARG REQS=requirements.txt
COPY requirements*.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ${REQS}

# Copiar el resto del proyecto
COPY . .

# Normalizar saltos de línea (por si se construye desde Windows) y permitir ejecución
RUN sed -i 's/\r$//' entrypoint.sh run_all.py && chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
