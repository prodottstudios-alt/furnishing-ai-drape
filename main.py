import os
import base64
import io
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image

app = FastAPI(title="Furnishing AI Precise Drape API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DrapeRequest(BaseModel):
    background: str  # base64
    objects: List[str]  # list of base64 patterns
    mode: str = "bedsheet"

def decode_image(b64_str: str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    img_np = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

def encode_image(img_np):
    _, buffer = cv2.imencode('.jpg', img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer).decode('utf-8')

from segment_anything import sam_model_registry, SamPredictor
import torch

# Load SAM Model
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

def download_model():
    if not os.path.exists(SAM_CHECKPOINT):
        print("Downloading SAM checkpoint (2.4GB)... this will take a while.")
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        import urllib.request
        urllib.request.urlretrieve(url, SAM_CHECKPOINT)
        print("Download complete.")

download_model()
print(f"Loading SAM model on {device}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM loaded successfully.")

def get_mask_sam(image_np, input_point, input_label):
    predictor.set_image(image_np)
    masks, scores, logits = predictor.predict(
        point_coords=np.array([input_point]),
        point_labels=np.array([input_label]),
        multimask_output=True,
    )
    # Pick the mask with highest score
    return masks[np.argmax(scores)]

def apply_texture(bg, pattern, mask):
    # 1. Resize pattern to fill the mask area roughly
    h, w = bg.shape[:2]
    pattern_resized = cv2.resize(pattern, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Simple Alpha Blending for verified 1:1 motif flow
    # (Future: Apply Displacement mapping from Grayscale BG)
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8) * 255
    
    # Invert mask to get the background
    bg_inv = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask.astype(np.uint8)*255))
    pattern_cut = cv2.bitwise_and(pattern_resized, pattern_resized, mask=mask.astype(np.uint8)*255)
    
    return cv2.add(bg_inv, pattern_cut)

@app.post("/drape")
async def drape_endpoint(req: DrapeRequest):
    try:
        bg = decode_image(req.background)
        patterns = [decode_image(obj) for obj in req.objects]
        
        if not patterns:
            raise HTTPException(status_code=400, detail="No patterns provided")

        # AUTO-SEGMENTATION:
        # For home-furnishing, center of the image is usually the bed.
        h, w = bg.shape[:2]
        center_point = [w // 2, h // 2] 
        
        # Get mask of the blanket at the center
        mask = get_mask_sam(bg, center_point, 1)
        
        # Apply pattern 1:1 using the mask
        result_img = apply_texture(bg, patterns[0], mask)
        
        return {"imageUrl": f"data:image/jpeg;base64,{encode_image(result_img)}"}
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
