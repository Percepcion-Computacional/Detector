from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from routes.websockets import router as websocket_router
import os

SNAPSHOTS_DIR = "snapshots"
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

app = FastAPI(
    title="Weapon Detection API",
    description="Backend API for real-time weapon detection.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket_router)

# Servir imágenes de snapshots como archivos estáticos
app.mount("/snapshots", StaticFiles(directory=SNAPSHOTS_DIR), name="snapshots")


@app.get("/")
def read_root():
    return {"message": "Weapon Detection API is running. Ready to receive frames."}


@app.get("/snapshots-list")
def list_snapshots():
    """Lista todos los snapshots guardados, ordenados del más reciente al más antiguo."""
    try:
        files = sorted(
            [f for f in os.listdir(SNAPSHOTS_DIR) if f.endswith(".jpg")],
            reverse=True
        )
        return {"snapshots": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/snapshots/{filename}")
def delete_snapshot(filename: str):
    """Elimina un snapshot por nombre de archivo."""
    # Sanitizar el nombre para evitar path traversal
    filename = os.path.basename(filename)
    filepath = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Snapshot not found")
    os.remove(filepath)
    return {"deleted": filename}
