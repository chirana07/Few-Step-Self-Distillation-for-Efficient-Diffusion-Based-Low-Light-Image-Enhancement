import os
import glob
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from config import Config
from model import ResidualConditionedUNet
from diffusion import DiffusionEngine


def pad_to_multiple_of_8(img):
    w, h = img.size
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8
    pad_w = new_w - w
    pad_h = new_h - h
    img = TF.pad(img, [0, 0, pad_w, pad_h], fill=0)
    return img, (w, h)


def main():
    conf = Config()

    test_samples_dir = "./test_samples"
    results_dir = "./test_results_20"
    checkpoint_path = "/Users/chirana/Desktop/UOM/Sem 3/Software project IFS/Diffusion_new/LUMIDIFF/New_dif/checkpoints/last_pth_only/final.pth"

    os.makedirs(results_dir, exist_ok=True)

    device = conf.DEVICE
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    use_prior = bool(checkpoint.get("use_illum_prior", False)) if isinstance(checkpoint, dict) else False

    model = ResidualConditionedUNet(use_illum_prior=use_prior).to(device)
    diff = DiffusionEngine()

    if "ema" in checkpoint:
        print("Loading EMA weights...")
        model.load_state_dict(checkpoint["ema"])
    elif "model" in checkpoint:
        print("Loading standard model weights...")
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    image_paths = sorted(
        glob.glob(os.path.join(test_samples_dir, "*.png")) +
        glob.glob(os.path.join(test_samples_dir, "*.jpg")) +
        glob.glob(os.path.join(test_samples_dir, "*.jpeg"))
    )

    if not image_paths:
        print(f"No images found in {test_samples_dir}")
        return

    print(f"Found {len(image_paths)} images")

    for img_path in tqdm(image_paths):
        basename = os.path.basename(img_path)

        low = Image.open(img_path).convert("RGB")
        low_padded, original_size = pad_to_multiple_of_8(low)

        low_tensor = (TF.to_tensor(low_padded) - 0.5) * 2.0
        low_tensor = low_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            gen_tensor = diff.sample(model, low_tensor)

        gen_tensor = (gen_tensor + 1.0) / 2.0
        gen_tensor = torch.clamp(gen_tensor, 0.0, 1.0)

        gen_img = TF.to_pil_image(gen_tensor.squeeze(0).cpu())
        gen_img = gen_img.crop((0, 0, original_size[0], original_size[1]))

        out_path = os.path.join(results_dir, f"enhanced_{basename}")
        gen_img.save(out_path)

    print(f"Inference complete. Results saved to {results_dir}")


if __name__ == "__main__":
    main()