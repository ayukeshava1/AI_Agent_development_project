from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pathlib import Path
import socketio
import os
import asyncio
from dotenv import load_dotenv  # New: Load .env
load_dotenv()  # Load keys

from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta
from supabase import create_client, Client

app = FastAPI(title="AI Converter Backend", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socketio (wrap AFTER app init)
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

# Auth
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv('SECRET_KEY', "your-secret-key-change-in-prod")  # .env or stub
ALGORITHM = "HS256"

# Supabase (optional—comment if no .env keys)
try:
    supabase: Client = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_ANON_KEY'))
except:
    supabase = None
    print("Supabase not loaded—check .env keys.")

@app.get("/")
def root():
    return {"message": "Backend alive! Ready for conversions."}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Stub user (real: Query Supabase users table)
    user = {"email": form_data.username, "hashed_pass": pwd_context.hash(form_data.password)}
    access_token = jwt.encode(
        {"sub": user["email"], "exp": datetime.utcnow() + timedelta(minutes=30)},
        SECRET_KEY, algorithm=ALGORITHM
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/signup")
async def signup(email: str = Form(...), password: str = Form(...)):
    if not supabase:
        return {"status": "error", "message": "Supabase not configured"}
    hashed = pwd_context.hash(password)
    data = {"email": email, "hashed_pass": hashed}
    res = supabase.table('users').insert(data).execute()
    return {"status": "user created" if res.data else "error", "data": res.data}

@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")

@app.post("/convert")
async def convert_file(
    file: UploadFile = File(...),
    mode: str = Form(...)
):
    job_id = "fake-job-" + str(hash(file.filename))
    temp_path = Path(f"./temp/{file.filename}")
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Stub progress (real: agent_pipeline)
    progress = [
        "Extracting content...",
        "Processing with AI models...",
        "Generating output...",
        "Finalizing file..."
    ]
    preview_url = f"http://localhost:8000/static/stub-preview-{mode.split('-')[0]}.pdf"
    download_url = f"http://localhost:8000/static/stub-download-{mode.split('-')[0]}.zip"
    limit_exceeded = False

    # Emit progress via Socketio BEFORE return
    for i, step in enumerate(progress):
        await sio.emit('progress_update', {'job_id': job_id, 'step': step, 'percent': (i+1)/len(progress)*100})
        await asyncio.sleep(1.5)  # Fake delay

    await sio.emit('complete', {'job_id': job_id, 'output': {"previewUrl": preview_url, "downloadUrl": download_url}})

    temp_path.unlink(missing_ok=True)

    return JSONResponse(content={
        "status": "success",
        "job_id": job_id,
        "progress": progress,
        "output": {"previewUrl": preview_url, "downloadUrl": download_url},
        "limit_exceeded": limit_exceeded
    })

@app.get("/user/files")
def get_user_files():
    mock_data = [
        {"name": "lecture.mp4", "type": "video", "thumb": "stub-thumb-video.jpg", "date": "2025-10-14", "size": "5MB"},
        {"name": "notes.pdf", "type": "pdf", "thumb": "stub-thumb-pdf.jpg", "date": "2025-10-13", "size": "2MB"},
        {"name": "conversion.zip", "type": "conversion", "thumb": "stub-thumb-zip.jpg", "date": "2025-10-12", "size": "3MB"}
    ]
    return {"files": mock_data}

@app.get("/user/gallery")
def get_user_gallery():
    return {
        "public_samples": [
            {"title": "Lecture to Notes", "thumb": "stub-gallery1.jpg", "before": "video", "after": "pdf"},
            {"title": "Report to Explainer", "thumb": "stub-gallery2.jpg", "before": "pdf", "after": "video"}
        ],
        "user_feed": []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)