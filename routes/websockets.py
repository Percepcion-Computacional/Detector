from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.inference import detector_service
import json
import asyncio

router = APIRouter()


@router.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket client connected.")
    try:
        while True:
            # receive_text lanzará WebSocketDisconnect si el cliente se va
            data = await websocket.receive_text()

            # Parse JSON payload
            try:
                payload = json.loads(data)
                base64_img = payload.get("image", "")
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON payload"})
                continue

            if not base64_img:
                await websocket.send_json({"error": "No image data provided"})
                continue

            # Run inference (async — won't block the event loop)
            result = await detector_service.process_frame(base64_img)

            # Send Pydantic model as JSON
            await websocket.send_text(result.model_dump_json())

    except WebSocketDisconnect:
        print("⚠️  WebSocket client disconnected.")
    except Exception as e:
        print(f"❌ Unexpected WebSocket error: {e}")
        # Solo intentar cerrar si el websocket aún está en estado abierto
        try:
            await websocket.close(code=1011)
        except Exception:
            pass  # Ya estaba cerrado, ignorar
