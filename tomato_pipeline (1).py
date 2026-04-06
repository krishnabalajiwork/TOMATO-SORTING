import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision import models


def build_classifier_model(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def load_classifier(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = build_classifier_model(2)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    else:
        model = checkpoint

    model.to(device)
    model.eval()
    return model


def make_transform(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def classify_crop(crop, model, transform, device, labels):
    # crop is expected as a PIL Image or numpy array (RGB)
    if not isinstance(crop, Image.Image):
        crop = Image.fromarray(crop)

    tensor = transform(crop).unsqueeze(0).to(device)

    out = model(tensor)

    prob = torch.softmax(out, dim=1)

    conf, idx = torch.max(prob, 1)

    return labels[idx.item()], float(conf.item())
