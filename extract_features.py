import torch
import clip
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import os

if __name__ == "__main__":
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    cifar100 = CIFAR100(root=os.path.join('img'), download=True, train=False, transform=preprocess)

    loader = DataLoader(
        cifar100, 
        batch_size=128, 
        shuffle=False, 
        num_workers=4
    )


    print("Extracting features, device: ", device)
    features = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, _ in loader:
            images = images.to(device)

            # Extract features
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            features.append(image_features.cpu())
    
    features = torch.cat(features, dim=0)
    os.makedirs('features', exist_ok=True)
    torch.save(features, os.path.join('features', 'cifar100_features.pt'))