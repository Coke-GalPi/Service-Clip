# Librerias de FastAPI
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import tempfile


# Librerias de Modelo Clip
import os
import clip
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.datasets import CIFAR10, CIFAR100

app = FastAPI(title="WebServiceClip",
            description="Hola, este es un servicio web que utiliza el modelo CLIP para predecir la clase de una imagen de CIFAR10 o CIFAR100.",
            version="1.0.0")

# Load the model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device)

# Load CIFAR10 and CIFAR100 datasets
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True)
text_inputs10 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)
templates10 = [f"a photo of a {c}" for c in cifar10.classes]

cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True)
text_inputs100 = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
templates100 = [f"a photo of a {c}" for c in cifar100.classes]

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

        # Calculate features for CIFAR10
        with torch.no_grad():
            image_features10 = model.encode_image(image_input)
            text_features10 = model.encode_text(text_inputs10)

        # Calculate features for CIFAR100
        with torch.no_grad():
            image_features100 = model.encode_image(image_input)
            text_features100 = model.encode_text(text_inputs100)

        # Calculate similarity for CIFAR10
        image_features10 /= image_features10.norm(dim=-1, keepdim=True)
        text_features10 /= text_features10.norm(dim=-1, keepdim=True)
        similarity10 = (100.0 * image_features10 @ text_features10.T).softmax(dim=-1)
        values10, indices10 = similarity10[0].topk(3)

        # Calculate similarity for CIFAR100
        image_features100 /= image_features100.norm(dim=-1, keepdim=True)
        text_features100 /= text_features100.norm(dim=-1, keepdim=True)
        similarity100 = (100.0 * image_features100 @ text_features100.T).softmax(dim=-1)
        values100, indices100 = similarity100[0].topk(3)

        # Select the dataset with the highest top-3 similarity
        if values10.max() >= values100.max():
            values = values10
            indices = indices10
            classes = cifar10.classes
            templates = templates10
        else:
            values = values100
            indices = indices100
            classes = cifar100.classes
            templates = templates100

        # Prepare the result
        result = {}
        if values[0].item() >= 0.9:
            result["prediction"] = templates[indices[0].item()]
        elif values[0].item() < 0.9 and values[0].item() >= 0.75:
            result["top_predictions"] = {classes[idx]: f"{100 * val.item():.2f}%" for val, idx in zip(values, indices)}
        else:
            result["message"] = "Cannot be predicted."

        return JSONResponse(content=result)