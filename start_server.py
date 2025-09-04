"""
Start FastAPI server without auto-reload for production
"""
import uvicorn
from fastapi_main import app

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Use different port
        reload=False  # Disable auto-reload for stability
    )
