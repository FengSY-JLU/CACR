# test_inference.py
import argparse
import torch
from src.model.network import RefinedColorCorrectionNet
from src.model.priors import default_priors
from PIL import Image
import torchvision.transforms as transforms
import os

def main():
    parser = argparse.ArgumentParser(description="Quick Inference Test")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/UIEBD_final/UIEBD_final.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--learned_priors', type=str, default='udcp,clahee,ruie',
                        help='Comma-separated priors (e.g. udcp,clahee,ruie)')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load priors
    learned_prior_cfgs = []
    prior_mapping = {
        "ruie": {"type": "restormer", "path": "", "name": "RUIE"},
        "woruie": {"type": "restormer", "path": "./src/model/restormer/best_restormer_prior_woruie.pth", "name": "woRUIE"},
    }

    for name in opt.learned_priors.split(','):
        name = name.strip().lower()
        if name in prior_mapping:
            cfg = prior_mapping[name]
            if cfg["path"] == "" or os.path.exists(cfg["path"]):
                learned_prior_cfgs.append(cfg)

    prior_modules = default_priors(device=device, learned_prior_cfgs=learned_prior_cfgs)
    
    # Load model
    model = RefinedColorCorrectionNet(prior_modules=prior_modules).to(device)
    
    print(f"Loading checkpoint: {opt.checkpoint}")
    checkpoint = torch.load(opt.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load and process image
    img = Image.open(opt.image).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(x=x)
        result = output["I_refined"].squeeze(0).cpu()

    # Save result
    result_img = transforms.ToPILImage()(result.clamp(0, 1))
    result_img.save(opt.output)
    print(f"✅ Inference completed! Result saved to: {opt.output}")

if __name__ == "__main__":
    main()