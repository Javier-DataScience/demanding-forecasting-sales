# Base image con Python
FROM python:3.11-slim

# Evitar warnings de buffering
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero para aprovechar cache
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Comando por defecto al correr el contenedor
CMD ["python", "src/models/train_model.py"]
