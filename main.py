from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
import clip
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import CIFAR100

app = FastAPI(title="WebServiceClip",
            description="Hola, este es un servicio web que utiliza el modelo CLIP para predecir la clase de una imagen de CIFAR10 o CIFAR100.",
            version="1.0.0")

# Load the model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)

# Load CIFAR100 datasets
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True)
text_inputs100 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
templates100 = [f"a photo of a {c}" for c in cifar100.classes]
classes100 = cifar100.classes

@app.get("/")
async def read_root():
    return FileResponse("upload.html")

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "input.png")

        # Save the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Load and preprocess the image
        image = Image.open(temp_file_path).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Calculate features for CIFAR100
        with torch.no_grad():
            image_features100 = model.encode_image(image_input)
            text_features100 = model.encode_text(text_inputs100)

        # Calculate similarity for CIFAR100
        image_features100 /= image_features100.norm(dim=-1, keepdim=True)
        text_features100 /= text_features100.norm(dim=-1, keepdim=True)
        similarity100 = (100.0 * image_features100 @ text_features100.T).softmax(dim=-1)
        values100, indices100 = similarity100[0].topk(3)

        # Prepare the result
        result = {}
        if values100[0].item() >= 0.9:
            result["prediction"] = templates100[indices100[0].item()]
        elif values100[0].item() < 0.9 and values100[0].item() >= 0.75:
            result["top_predictions"] = {classes100[idx]: f"{100 * val.item():.2f}%" for val, idx in zip(values100, indices100)}
        else:
            result["message"] = "Cannot be predicted."

        return JSONResponse(content=result)