import torch
from models.arcface import ArcFaceRecognizer
from pathlib import Path

def main():
    out_dir = Path('weights')
    out_dir.mkdir(exist_ok=True)
    model = ArcFaceRecognizer()
    weight_path = out_dir / 'arcface_iresnet100_dummy.pth'
    torch.save(model.backbone.state_dict(), weight_path)
    print(f'Dummy weights saved to {weight_path}')

if __name__ == '__main__':
    main()
