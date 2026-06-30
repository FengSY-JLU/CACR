# train.py (updated to match new losses.py logic)

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.model.network import RefinedColorCorrectionNet
from src.dataset.uieb_dataset import UIEBDataset
from src.model.losses import compute_total_loss
from src.model.pir.pir_fusion import ProgressiveInputRefiner  # 新增模块导入

def train():
    # === Configs ===
    epochs = 10
    batch_size = 1
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    dataset = UIEBDataset(split='train/image')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RefinedColorCorrectionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pir_module = ProgressiveInputRefiner(debug_dir="debug/pir")  # 实例化模块

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        a1_total, a2_total, a3_total = 0.0, 0.0, 0.0
        pixel_count = 0

        for idx, (batch, filename) in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1}]")):
            batch = batch.to(device)

            # Forward Pass
            I_refined, I_synthesis, attns, ts, I_phys = model(batch)
            attn1, attn2 = attns
            A1, A2, A3 = attn1
            del I_synthesis, ts, attn2  # 释放不必要的变量以节省内存
            torch.cuda.empty_cache()

            # === Attention stats ===
            a1_total += A1.sum().item()
            a2_total += A2.sum().item()
            a3_total += A3.sum().item()
            pixel_count += A1.numel()

            # === Compute PIR input ===
            pir_input = pir_module(batch, I_refined.detach(), filename[0], epoch)

            # Forward again with PIR input
            I_refined_pir, I_synth_pir, attns_pir, ts_pir, I_phys_pir = model(pir_input)

            # === Compute Loss ===
            loss, loss_dict = compute_total_loss(
                I_input=batch,
                I_refined=I_refined,
                I_synthesis_pir=I_synth_pir,
                I_refined_pir=I_refined_pir,
                I_phys_pir=I_phys_pir,
                I_phys_input=I_phys,
                loss_proxy_t=None  # 可加入t一致性项
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # === Log ===
        avg_a1 = a1_total / pixel_count
        avg_a2 = a2_total / pixel_count
        avg_a3 = a3_total / pixel_count

        print(f"[Attention Weights] A1_gray: {avg_a1:.4f}, A2_red: {avg_a2:.4f}, A3_phys: {avg_a3:.4f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Total Loss = {avg_loss:.4f}")
        print(f"  Proxy Red: {loss_dict['stage2_red_consistency']:.4f} | Proxy Cyan: {loss_dict['stage2_cyan_consistency']:.4f}")
        print(f"  Red Boost Loss: {loss_dict['stage1_red_boost']:.4f} | Brightness: {loss_dict['stage2_brightness']:.4f}")
        print(f"  Physics Proxy Loss: {loss_dict['physics_proxy_loss']:.4f} | t Consistency: {loss_dict['transmission_consistency_proxy']:.4f}")

        # Save
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[✓] Model saved to {save_path}")


if __name__ == '__main__':
    train()
