from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from video_stream import VideoCamera
import torch

app = FastAPI()
camera = VideoCamera("cvtest.avi")

print("CUDA Available:", torch.cuda.is_available())

def generate_frames():
    while True:
        frame, _ = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/status")
async def get_status():
    _, status = camera.get_frame()
    return {"status": status}



app.mount("/", StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame"))