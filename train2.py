# train.py with PIR structure-aware pseudo input

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
    pir_module = ProgressiveInputRefiner(debug_dir="debug/pir")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        a1_total, a2_total, a3_total = 0.0, 0.0, 0.0
        pixel_count = 0

        for idx, (batch, filename) in enumerate(tqdm(dataloader, desc=f"[Epoch {epoch+1}]")):
            batch = batch.to(device)

            # === 主分支前向传播 ===
            I_refined, I_synthesis, attns, ts, I_phys = model(batch)
            attn1, attn2 = attns
            t1, t2 = ts
            A1, A2, A3 = attn1

            a1_total += A1.sum().item()
            a2_total += A2.sum().item()
            a3_total += A3.sum().item()
            pixel_count += A1.numel()

            # === PIR 输入生成 ===
            pir_input = pir_module(batch, I_refined.detach(), filename[0], epoch)

            # === PIR 分支前向传播（不参与反向传播）===
            with torch.no_grad():
                I_refined_pir, I_synth_pir, attns_pir, ts_pir, I_phys_pir = model(pir_input)

            # === 损失计算 ===
            loss_main, loss_dict_main = compute_total_loss(batch, I_synthesis, I_refined, t1, t2, I_phys)
            loss_pir, loss_dict_pir = compute_total_loss(batch, I_synth_pir, I_refined_pir, ts_pir[0], ts_pir[1], I_phys_pir)

            total_loss = loss_main + 0.2 * loss_pir  # PIR 一致性损失权重

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # === 日志打印 ===
        avg_a1 = a1_total / pixel_count
        avg_a2 = a2_total / pixel_count
        avg_a3 = a3_total / pixel_count
        avg_loss = epoch_loss / len(dataloader)

        print(f"\n[Epoch {epoch+1}] Avg Total Loss = {avg_loss:.4f}")
        print(f"[Attention] A1_gray: {avg_a1:.4f} | A2_red: {avg_a2:.4f} | A3_phys: {avg_a3:.4f}")
        print(f"[Stage1]  L1: {loss_dict_main['stage1_l1']:.4f} | SSIM: {loss_dict_main['stage1_ssim']:.4f} | Hist: {loss_dict_main['stage1_hist']:.4f} | PhysRecon: {loss_dict_main['stage1_phys_recon']:.4f} | RedBoost: {loss_dict_main['stage1_red_boost']:.4f}")
        print(f"[Stage2]  L1: {loss_dict_main['stage2_l1']:.4f} | SSIM: {loss_dict_main['stage2_ssim']:.4f} | Bright: {loss_dict_main['stage2_bright']:.4f} | Consistency: {loss_dict_main['stage2_consistency']:.4f}")
        print(f"[Proxy]   Red: {loss_dict_main['stage2_red_consistency']:.4f} | Cyan: {loss_dict_main['stage2_cyan_consistency']:.4f}")
        print(f"[Physics] Transmission Consistency: {loss_dict_main['t_consistency']:.4f}")

        # === 模型保存 ===
        save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[✓] Model saved to {save_path}")


if __name__ == '__main__':
    train()
