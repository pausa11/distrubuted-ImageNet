import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

# Add src to path to import imagenet_classes
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
try:
    from imagenet_classes import IMAGENET_SYNSETS, IMAGENET_CLASS_NAMES
except ImportError:
    # Fallback if not found, though it should be there
    IMAGENET_SYNSETS = []
    IMAGENET_CLASS_NAMES = {}

def build_model(num_classes: int = 1000) -> nn.Module:
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def predict(image_path, checkpoint_path, device='cpu'):
    # Define transforms (standard ImageNet transforms)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

    # Load model
    try:
        model = build_model(len(IMAGENET_SYNSETS) if IMAGENET_SYNSETS else 1000)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 1
        top_prob, top_catid = torch.topk(probabilities, 1)
        class_idx = top_catid[0].item()
        confidence = top_prob[0].item()
        
        synset = IMAGENET_SYNSETS[class_idx] if class_idx < len(IMAGENET_SYNSETS) else "Unknown"
        class_name = IMAGENET_CLASS_NAMES.get(synset, "Unknown")

        return {
            "class_index": class_idx,
            "synset": synset,
            "class_name": class_name,
            "confidence": confidence
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to the image file")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = predict(args.image_path, args.checkpoint_path, device)
    print(json.dumps(result))
