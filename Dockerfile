# Usar una imagen oficial de Python ligera
FROM python:3.11-slim

# Evitar que Python escriba archivos .pyc y forzar salida no bufferizada
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema requeridas por OpenCV y Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requerimientos y cachear la capa de instalación
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del proyecto al contenedor
COPY . .

# Exponer el puerto donde corre FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación en producción
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
