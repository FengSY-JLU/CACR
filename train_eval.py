# train_eval.py
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
from src.model.losses3 import compute_total_loss, contribution_alignment_loss
from src.model.priors import default_priors
from evaluation_fast import evaluate
from src.dataset.data import get_training_set, get_eval_set

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

parser = argparse.ArgumentParser()
parser.add_argument('--nEpochs', type=int, default=200)
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--data_train', type=str, default='./Dataset/UIE/UIEBD/train/image')
parser.add_argument('--label_train', type=str, default='./Dataset/UIE/UIEBD/train/label')
parser.add_argument('--data_test', type=str, default='./Dataset/UIE/UIEBD/test/image')
parser.add_argument('--label_test', type=str, default='./Dataset/UIE/UIEBD/test/label')
parser.add_argument('--indicator', type=str, default='UIEBD_contributor_0603_manifold_woopt_ablation_unsharp')
parser.add_argument('--output_folder', default='Results_contributor_0603_manifold_woopt_ablation_unsharp/')
# parser.add_argument('--data_train', type=str, default='./Dataset/UIE/EUVP/Paired/underwater_imagenet/trainA')
# parser.add_argument('--label_train', type=str, default='./Dataset/UIE/EUVP/Paired/underwater_imagenet/trainB')
# parser.add_argument('--data_test', type=str, default='./Dataset/UIE/EUVP/Paired/underwater_imagenet/validation')
# parser.add_argument('--label_test', type=str, default='./Dataset/UIE/EUVP/Paired/underwater_imagenet/validation')
# parser.add_argument('--indicator', type=str, default='EUVP_')
# parser.add_argument('--output_folder', default='Results_euvp/')
# parser.add_argument('--data_train', type=str, default='./Dataset/UIE/LSUI/input')
# parser.add_argument('--label_train', type=str, default='./Dataset/UIE/LSUI/GT')
# parser.add_argument('--data_test', type=str, default='./Dataset/UIE/LSUI/input_test')
# parser.add_argument('--label_test', type=str, default='./Dataset/UIE/LSUI/GT_test')
# parser.add_argument('--indicator', type=str, default='lsui_')
# parser.add_argument('--output_folder', default='Results_lsui/')
# parser.add_argument('--data_train', type=str, default='./Dataset/UIE/RUIE/train')
# parser.add_argument('--label_train', type=str, default='./Dataset/UIE/RUIE/train')
# parser.add_argument('--data_test', type=str, default='./Dataset/UIE/RUIE/test')
# parser.add_argument('--label_test', type=str, default='./Dataset/UIE/RUIE/test')
# parser.add_argument('--indicator', type=str, default='RUIE_manifold_hdp_pool')
# parser.add_argument('--output_folder', default='Results_RUIE_manifold_hdp_pool/')
# parser.add_argument('--data_train', type=str, default='./Dataset/UIE/OceanDark')
# parser.add_argument('--label_train', type=str, default='./Dataset/UIE/OceanDark')
# parser.add_argument('--data_test', type=str, default='./Dataset/UIE/OceanDark')
# parser.add_argument('--label_test', type=str, default='./Dataset/UIE/OceanDark')
# parser.add_argument('--indicator', type=str, default='OceanDark_contributor')
# parser.add_argument('--output_folder', default='Results_OceanDark_contributor')
parser.add_argument('--restormer_prior_path', type=str, default='')
parser.add_argument('--restormer_prior_path_woruie', type=str, default='./src/model/restormer/best_restormer_prior_woruie.pth')
opt = parser.parse_args()

set_seed(opt.seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

learned_prior_cfgs = []

if os.path.exists(opt.restormer_prior_path):

    learned_prior_cfgs.append(
        {
            "type":"restormer",
            "path":opt.restormer_prior_path,
            "name":"RUIE"
        }
    )

if os.path.exists(
    opt.restormer_prior_path_woruie
):

    learned_prior_cfgs.append(
        {
            "type":"restormer",

            "path":opt.restormer_prior_path_woruie,

            "name":"woRUIE"
        }
    )

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
test_set = get_eval_set(opt.data_test, opt.label_test)

train_loader = DataLoader(
    train_set,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=opt.threads,
    pin_memory=False
)

test_loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

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

        if opt.debug == True:
            if epoch < 5:
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
                # print(f"contribution_loss: ",{loss_offsets_align})

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
                "restormer_prior_path": opt.restormer_prior_path,
                "device": str(device),
                "cudnn_enabled": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "data": {
                "data_test": opt.data_test,
                "label_test": opt.label_test,
            }
        }

        torch.save(save_obj, save_path)

        print(f"[✓] Saved: {save_path}")

        if epoch % 200 == 0:

            model.eval()

            if hasattr(model.stage2, "como"):
                model.stage2.como.lambda_prune = save_obj["runtime"]["lambda_prune"]

            evaluate(model, opt, test_loader, epoch)

            model.train()