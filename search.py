import torch
import clip
import os
from torchvision.datasets import CIFAR100
from PIL import Image

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    img_features = torch.load('features/cifar100_features.pt').to(device)

    cifar100 = CIFAR100(root=os.path.join('img'), download=True, train=False)
    cifar100 = cifar100.data

    while True:
        query = input('Query > ')
        print(f'Query: {query}')

        text = torch.cat([clip.tokenize(query)]).to(device)

        with torch.no_grad():
            text_feature = model.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

        dists = torch.cdist(text_feature, img_features)
        top5 = torch.topk(-dists.view(-1), k=5).indices

        out_dir = 'out'
        os.makedirs(out_dir, exist_ok=True)

        # numpy→PIL で保存
        for rank, idx in enumerate(top5):
            arr = cifar100[idx] # (32,32,3), uint8
            img = Image.fromarray(arr)
            img.save(os.path.join(out_dir, f"{rank}.png"))
            print(f"#{rank+1}: index={idx} → {out_dir}/{rank}.png")