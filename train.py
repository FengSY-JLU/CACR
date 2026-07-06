# train.py
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from src.model.network import RefinedColorCorrectionNet
from src.model.losses import compute_total_loss
from src.model.priors import default_priors
from src.dataset.data import get_training_set

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_runtime_state(model):
    state = {}
    if hasattr(model, "stage2") and hasattr(model.stage2, "como"):
        state["lambda_prune"] = float(getattr(model.stage2.como, "lambda_prune", 0.0))
    return state

def pruning_schedule(epoch, max_epoch, start=0.0, end=0.05, warmup_ratio=0.2):
    warmup_epochs = int(max_epoch * warmup_ratio)
    if epoch <= warmup_epochs:
        return start
    t = (epoch - warmup_epochs) / max(1, max_epoch - warmup_epochs)
    return start + t * (end - start)

parser = argparse.ArgumentParser(description="Pure Training Script for RefinedColorCorrectionNet")
parser.add_argument('--nEpochs', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--batchSize', type=int, default=1, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--threads', type=int, default=4, help='Number of data loader threads')
parser.add_argument('--patch_size', type=int, default=128, help='Patch size for training')
parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--data_train', type=str, default='./Dataset/UIE/UIEBD/train/image', help='Path to training images')
parser.add_argument('--label_train', type=str, default='./Dataset/UIE/UIEBD/train/label', help='Path to training labels')
parser.add_argument('--indicator', type=str, default='UIEBD_contributor_0603_manifold_woopt_ablation_unsharp', help='Experiment indicator')
parser.add_argument('--learned_priors', type=str, default='ruie,woruie',
                    help='Comma-separated list of learned priors to use (e.g. ruie,woruie,all)')

parser.add_argument('--restormer_prior_path', type=str, default='', help='Path to restormer prior (RUIE)')
parser.add_argument('--restormer_prior_path_woruie', type=str, default='./src/model/restormer/best_restormer_prior_woruie.pth', help='Path to restormer prior without RUIE')

opt = parser.parse_args()

set_seed(opt.seed)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ====================== Prior ======================
learned_prior_cfgs = []

prior_mapping = {
    "ruie": {
        "type": "restormer",
        "path": opt.restormer_prior_path,
        "name": "RUIE"
    },
    "woruie": {
        "type": "restormer",
        "path": opt.restormer_prior_path_woruie,
        "name": "woRUIE"
    }
}

selected = [p.strip().lower() for p in opt.learned_priors.split(',')]

if "all" in selected:
    selected = ["ruie", "woruie"]

for name in selected:
    if name in prior_mapping:
        cfg = prior_mapping[name]
        if os.path.exists(cfg["path"]):
            learned_prior_cfgs.append(cfg)
        else:
            print(f"Warning: Prior {name} path not found: {cfg['path']}")
    else:
        print(f"Warning: Unknown prior '{name}', skipping.")

print(f"Using learned priors: {[cfg['name'] for cfg in learned_prior_cfgs]}")

prior_modules = default_priors(
    device=device,
    learned_prior_cfgs=learned_prior_cfgs
)

model = RefinedColorCorrectionNet(prior_modules=prior_modules).to(device)

save_dir = f"checkpoints/{opt.indicator}/"
os.makedirs(save_dir, exist_ok=True)
debug_log_path = os.path.join(save_dir, "debug_loss.log")

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = lrs.MultiStepLR(optimizer, milestones=[100], gamma=0.5)

train_set = get_training_set(opt.data_train, opt.label_train, opt.patch_size, True)
train_loader = DataLoader(
    train_set,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=opt.threads,
    pin_memory=False
)

# ==================== train  ====================
def train(epoch):
    model.train()
    epoch_loss = 0
    epoch_loss_dict = defaultdict(float)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for iteration, batch in enumerate(pbar, 1):
        x, y = batch[0].to(device), batch[1].to(device)
        
        output = model(x=x)
        I_refined = output["I_refined"]
        I_synthesis = output["I_synthesis"]
        prior_list = output["priors"]
        weighted_offsets = output["offsets_stage2"]
        J = output["priors"][-1]
        como_out = output["como_out"]
        
        if opt.debug and epoch < 5:
            recon = como_out.get("recon", None)
            dw_out = como_out.get("dw_out", None)
            lambda_map = como_out.get("lambda_map", None)
            if recon is not None and dw_out is not None and lambda_map is not None:
                print(
                    f"[DEBUG Mamba scale] "
                    f"recon min/max: {recon.min().item():.4f}/{recon.max().item():.4f}, "
                    f"dw_out min/max: {dw_out.min().item():.4f}/{dw_out.max().item():.4f}, "
                    f"lambda_map min/max: {lambda_map.min().item():.4f}/{lambda_map.max().item():.4f}"
                )
        
        loss, loss_dict = compute_total_loss(
            I_input=x,
            I_refined=I_refined,
            I_synthesis=I_synthesis,
            priors=prior_list,
            weighted_offsets=weighted_offsets,
            cas_weights=model.stage2.last_cas_weights,
            attn_offset_only=como_out["att_offset_only"],
            J_phys=J,
            model=model,
            epoch=epoch,
            iteration=iteration,
        )
        
        epoch_loss += loss.item()
        for k, v in loss_dict.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            epoch_loss_dict[k] += val
        
        log_line = f"{epoch},{iteration}"
        for k, v in loss_dict.items():
            val = v.item() if torch.is_tensor(v) else v
            log_line += f",{k}={val:.6f}"
        with open(debug_log_path, "a") as f:
            f.write(log_line + "\n")
        
        if iteration % 100 == 0:
            print(f"\n[Epoch {epoch} | Iter {iteration}]")
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                print(f"{k:20s}: {val:.6f}")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = epoch_loss / len(train_loader)
    avg_loss_dict = {k: v / len(train_loader) for k, v in epoch_loss_dict.items()}
    
    print(f"\nEpoch {epoch} | Avg Loss: {avg_loss:.6f}")
    for k, v in avg_loss_dict.items():
        print(f"{k:20s}: {v:.6f}")
    
    return avg_loss

if __name__ == '__main__':
    for epoch in range(1, opt.nEpochs + 1):
        if hasattr(model.stage2, "como"):
            model.stage2.como.lambda_prune = pruning_schedule(epoch, opt.nEpochs)
        
        train(epoch)
        scheduler.step()
        
        save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        save_obj = {
            "state_dict": model.state_dict(),
            "runtime": get_runtime_state(model),
            "config": {
                "seed": opt.seed,
                "learned_priors": opt.learned_priors,
                "device": str(device),
            },
            "data": {
                "data_train": opt.data_train,
                "label_train": opt.label_train,
            }
        }
        torch.save(save_obj, save_path)
        print(f"[✓] Saved checkpoint: {save_path}")