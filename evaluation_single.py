# evaluation_single.py
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from src.model.network import RefinedColorCorrectionNet
from src.model.prior import default_priors, RestormerPrior
from src.dataset.data import get_eval_set
from evaluation_fast import evaluate

def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="checkpoints/UIEBD_contributor_0603_manifold_all_woopt/model_epoch_200.pth")
    parser.add_argument("--data_test", type=str, default='./Dataset/UIE/UIEBD/test/image')
    parser.add_argument("--label_test", type=str, default='./Dataset/UIE/UIEBD/test/label')
    parser.add_argument("--indicator", type=str, default="UIEBD_contributor_0603_manifold_all_woopt")
    parser.add_argument("--output_folder", type=str, default="Results_contributor_0603_manifold_all_woopt/")

    args = parser.parse_args()

    ckpt = torch.load(args.model_path, map_location=device)

    config = ckpt.get("config", {})

    set_seed(config.get("seed", 3407))

    torch.backends.cudnn.enabled = config.get("cudnn_enabled", False)
    torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", False)
    torch.backends.cudnn.deterministic = config.get("cudnn_deterministic", True)

    restormer_path = config.get(
        "restormer_prior_path",
        "./src/model/restormer/best_restormer_prior.pth"
    )

    priors = default_priors(device=device)

    if os.path.exists(restormer_path):
        priors.append(RestormerPrior(restormer_path, device))

    model = RefinedColorCorrectionNet(prior_modules=priors).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)

    runtime = ckpt.get("runtime", {})

    if hasattr(model.stage2, "como"):
        model.stage2.como.lambda_prune = runtime.get("lambda_prune", 0.0)

    print(f"[✓] lambda_prune = {model.stage2.como.lambda_prune}")

    data_cfg = ckpt.get("data", {})

    if args.data_test is None:
        args.data_test = data_cfg.get("data_test")

    if args.label_test is None:
        args.label_test = data_cfg.get("label_test")

    print(f"[✓] data_test = {args.data_test}")
    print(f"[✓] label_test = {args.label_test}")

    test_set = get_eval_set(args.data_test, args.label_test)

    loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    class Opt:
        pass

    opt = Opt()
    opt.output_folder = args.output_folder
    opt.indicator = args.indicator

    model.eval()

    evaluate(model, opt, loader, epoch="standalone")