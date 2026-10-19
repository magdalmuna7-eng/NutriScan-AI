"""Carga exactamente el modelo y las clases exportados por training.py."""
import json
from pathlib import Path
import torch
from torchvision.models import resnet18, ResNet18_Weights


def trained_bundle(root):
    root = Path(root)
    config = root / 'deployment.json'
    if not config.exists():
        return None, None
    metadata = json.loads(config.read_text())
    if metadata['selected'] == 'clip':
        return metadata, None
    if metadata['selected'] != 'resnet18':
        raise ValueError('Modelo exportado desconocido')
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(metadata['classes']))
    model.load_state_dict(torch.load(root / 'resnet18.pt', map_location='cpu', weights_only=True))
    return metadata, model.eval()


def predict_resnet(model, image):
    x = ResNet18_Weights.DEFAULT.transforms()(image).unsqueeze(0)
    with torch.inference_mode():
        return model(x).softmax(-1)[0].tolist()
