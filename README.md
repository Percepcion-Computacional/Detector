# Weapon Detection API 🛡️

Backend API para la detección de armas en tiempo real mediante visión por computadora. Este servicio procesa frames de video, ejecuta inferencia utilizando YOLOv8 y devuelve las detecciones utilizando WebSockets para latencia ultra baja. También provee endpoints REST para la gestión de snapshots (capturas de detecciones).

## 🚀 Tecnologías Principales

- **FastAPI**: Framework web rápido para la construcción de APIs.
- **WebSockets**: Comunicación bidireccional en tiempo real.
- **YOLOv8 (Ultralytics)**: Modelo de detección de objetos de última generación.
- **OpenCV**: Procesamiento y manipulación de imágenes.
- **PyTorch**: Framework de Deep Learning subyacente.
- **Uvicorn**: Servidor ASGI de alto rendimiento.

## 📂 Estructura del Proyecto

```text
detector/
├── main.py              # Punto de entrada de la aplicación FastAPI
├── models/              # Almacenamiento de pesos del modelo (ej. best.pt) y schemas
├── routes/              # Controladores de rutas (REST y WebSockets)
├── services/            # Lógica de negocio (ej. motor de inferencia)
├── utils/               # Funciones utilitarias (procesamiento de imágenes)
├── snapshots/           # Directorio auto-generado para imágenes detectadas
├── Dockerfile           # Configuración para despliegue en contenedor
├── requirements.txt     # Dependencias de Python
└── .gitignore           # Exclusiones de control de versiones
```

## 🛠️ Instalación y Ejecución Local

### Prerrequisitos
- Python 3.11+
- pip (gestor de paquetes de Python)

### Pasos

1. Clona el repositorio e ingresa al directorio del backend:
   ```bash
   git clone https://github.com/Percepcion-Computacional/Detector.git
   cd Detector/detector
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecuta el servidor:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   *La documentación interactiva de Swagger estará disponible en `http://localhost:8000/docs`.*

## 🐳 Ejecución con Docker

Para un despliegue rápido y consistente:

1. Construye la imagen de Docker:
   ```bash
   docker build -t weapon-detection-api .
   ```

2. Ejecuta el contenedor:
   ```bash
   docker run -p 8000:8000 weapon-detection-api
   ```

## 📡 Referencia de la API

### Endpoints REST

- **`GET /`**
  Ruta de prueba (Health Check) para confirmar que la API está funcionando.
- **`GET /snapshots-list`**
  Devuelve una lista paginada/ordenada de todos los snapshots guardados (detecciones positivas).
- **`DELETE /snapshots/{filename}`**
  Elimina un snapshot específico por su nombre de archivo.
- **`GET /snapshots/{filename}`**
  Ruta estática para servir las imágenes generadas.

### WebSockets

- **Conexión Principal (Ver `routes/websockets.py`)**
  Permite streaming bidireccional. El cliente envía frames codificados y el servidor responde con un JSON que contiene las cajas delimitadoras (bounding boxes), las clases detectadas y el nivel de confianza de la inferencia.

## 📄 Licencia

Consulte el archivo de licencia en la raíz del repositorio de GitHub para los términos de uso y distribución.
