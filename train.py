import os
import csv
import yaml
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.datasets import NightRemovalDataset
from models.DG_Mamba import DGMamba
from losses.losses import CombinedLoss
from utils.utils import calculate_psnr_ssim


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_curves(csv_path: str, out_dir: str):
    """
    Read CSV and draw curves (Loss/PSNR/SSIM/LR).
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Loss curve
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_loss.png"))
    plt.close()

    # PSNR curve
    plt.figure()
    plt.plot(df["epoch"], df["val_psnr"], label="val_psnr")
    plt.plot(df["epoch"], df["best_psnr_so_far"], label="best_psnr_so_far")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_psnr.png"))
    plt.close()

    # SSIM curve
    plt.figure()
    plt.plot(df["epoch"], df["val_ssim"], label="val_ssim")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "curve_ssim.png"))
    plt.close()

    # LR curve
    if "lr" in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df["lr"], label="lr")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "curve_lr.png"))
        plt.close()


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    for real_shadow, real_free in val_loader:
        real_shadow = real_shadow.to(device)
        real_free = real_free.to(device)

        pred_free = model(real_shadow)
        loss = criterion(pred_free, real_free)
        val_loss += loss.item()

        psnr_value, ssim_value = calculate_psnr_ssim(pred_free, real_free)
        val_psnr += psnr_value
        val_ssim += ssim_value

    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = val_psnr / len(val_loader)
    avg_val_ssim = val_ssim / len(val_loader)
    return avg_val_loss, avg_val_psnr, avg_val_ssim


def append_row_csv(csv_path, fieldnames, row_dict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()


def train(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    image_size = config["image_size"]

    # Output dirs
    save_dir = config.get("save_dir", "checkpoints/")
    output_dir = config.get("output_dir", "results/")
    ensure_dir(save_dir)
    ensure_dir(output_dir)

    # Where to store logs/plots
    run_name = config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = os.path.join(output_dir, run_name)
    curves_dir = os.path.join(run_dir, "curves")
    ensure_dir(run_dir)
    ensure_dir(curves_dir)

    csv_path = os.path.join(run_dir, "training_log.csv")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = NightRemovalDataset(
        config["train_gt_dir"], config["train_lq_dir"], transform=transform, augment=True
    )
    val_dataset = NightRemovalDataset(
        config["val_gt_dir"], config["val_lq_dir"], transform=transform, augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # Model
    model = DGMamba(**config["model"]).to(device)

    # Optional: load weights
    weights_path = config.get("weights_path", None)
    if weights_path:
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print(f"[INFO] Loaded weights from: {weights_path}")
        else:
            print(f"[WARN] weights_path not found: {weights_path}")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"], eta_min=1e-8)
    criterion = CombinedLoss(device=device)

    best_psnr = float("-inf")
    best_epoch = -1
    best_path = os.path.join(save_dir, "best_model_psnr.pth")

    # CSV columns
    fieldnames = [
        "epoch", "train_loss", "val_loss", "val_psnr", "val_ssim",
        "lr", "best_psnr_so_far", "best_epoch"
    ]

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")
        for real_shadow, real_free in pbar:
            real_shadow = real_shadow.to(device, non_blocking=True)
            real_free = real_free.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_free = model(real_shadow)
            loss = criterion(pred_free, real_free)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # Validate
        avg_val_loss, avg_val_psnr, avg_val_ssim = validate(model, val_loader, criterion, device)

        # Best-by-PSNR save
        improved = avg_val_psnr > best_psnr
        if improved:
            best_psnr = avg_val_psnr
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] Epoch {best_epoch}: PSNR={best_psnr:.4f}  -> saved {best_path}")

        # Step scheduler AFTER validation (either is fine, just be consistent)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch+1}/{config['num_epochs']}]: "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
            f"PSNR={avg_val_psnr:.4f} | SSIM={avg_val_ssim:.4f} | lr={current_lr:.2e}"
        )

        # Write CSV every epoch (safe if job stops)
        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "lr": current_lr,
            "best_psnr_so_far": best_psnr,
            "best_epoch": best_epoch,
        }
        append_row_csv(csv_path, fieldnames, row)

        # Update plots every epoch (you can change to every N epochs)
        try:
            save_curves(csv_path, curves_dir)
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

    print(f"\n[DONE] Best PSNR = {best_psnr:.4f} at epoch {best_epoch}")
    print(f"[DONE] Best checkpoint: {best_path}")
    print(f"[DONE] CSV log: {csv_path}")
    print(f"[DONE] Curves folder: {curves_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
