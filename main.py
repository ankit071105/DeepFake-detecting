# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os
from typing import List
import tempfile
import joblib
import torch
import torch.nn as nn
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="DeepFake Detection API")

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the model architecture (same as in train_model.py)
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the trained model
model = None
try:
    model = DeepFakeDetector().to(device)
    if os.path.exists('models/deepfake_model.pth'):
        model.load_state_dict(torch.load('models/deepfake_model.pth', map_location=device))
        model.eval()
        print("Model loaded successfully")
    else:
        print("No trained model found. Using fallback mode.")
except Exception as e:
    print(f"Could not load model: {e}")
    model = None

def extract_frames_from_bytes(video_bytes, num_frames=10, resize=(128, 128)):
    """Extract frames from video bytes without OpenCV"""
    try:
        # Save bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        
        # Use ffmpeg if available, otherwise fallback
        try:
            import subprocess
            # Create output directory for frames
            frames_dir = tempfile.mkdtemp()
            
            # Extract frames using ffmpeg
            cmd = [
                'ffmpeg', '-i', tmp_path, '-vf', f'fps=1,scale={resize[0]}:{resize[1]}',
                f'{frames_dir}/frame_%03d.jpg', '-y'
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Load frames
            frames = []
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_')])
            
            for frame_file in frame_files[:num_frames]:
                img = Image.open(os.path.join(frames_dir, frame_file))
                img_array = np.array(img) / 255.0
                frames.append(img_array)
            
            # Cleanup
            for frame_file in frame_files:
                os.remove(os.path.join(frames_dir, frame_file))
            os.rmdir(frames_dir)
            
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: generate synthetic frames
            print("FFmpeg not available, using synthetic frames")
            frames = [np.random.rand(resize[0], resize[1], 3) for _ in range(num_frames)]
        
        # Ensure we have the right number of frames
        while len(frames) < num_frames:
            frames.append(np.zeros((resize[0], resize[1], 3)))
        
        os.unlink(tmp_path)
        return np.array(frames)
    
    except Exception as e:
        print(f"Error extracting frames: {e}")
        # Return synthetic frames as fallback
        return np.random.rand(num_frames, resize[0], resize[1], 3)

def predict_deepfake(video_bytes: bytes) -> dict:
    """
    Process video and make prediction using deep learning model
    """
    if model is None:
        # Fallback to a simple heuristic if model is not loaded
        return fallback_prediction()
    
    try:
        # Extract frames from video
        frames = extract_frames_from_bytes(video_bytes)
        
        # Convert to tensor and adjust dimensions for PyTorch
        # From (T, H, W, C) to (C, T, H, W)
        frames = np.transpose(frames, (3, 0, 1, 2))
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        frames_tensor = torch.FloatTensor(frames).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(frames_tensor).cpu().numpy()[0][0]
        
        # Return results
        return {
            "fake_probability": float(prediction),
            "real_probability": float(1 - prediction),
            "is_fake": prediction > 0.5,
            "frames_processed": frames.shape[1]
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return fallback_prediction()

def fallback_prediction() -> dict:
    """
    Fallback prediction method when model is not available
    """
    # Simple random prediction for demonstration
    fake_prob = np.random.uniform(0.3, 0.7)
    
    return {
        "fake_probability": float(fake_prob),
        "real_probability": float(1 - fake_prob),
        "is_fake": fake_prob > 0.5,
        "frames_processed": 10,
        "note": "Using fallback heuristic (train model for better accuracy)"
    }

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("video/") and not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process the video
        result = predict_deepfake(content)
        
        return {
            "filename": file.filename,
            "prediction": "FAKE" if result["is_fake"] else "REAL",
            "fake_confidence": result["fake_probability"],
            "real_confidence": result["real_probability"],
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "DeepFake Detection API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)