import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, Subset
from dataset import build_dataset_from_csv_files, BatteryPINNDataset
from model import MM_Q3D_KAN_PINN
from train import train_one_epoch, validate


def set_global_seed(seed, deterministic=False):
    """Set global RNG seeds for reproducible training/evaluation behavior."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _compute_binned_error_curves(y_true, y_pred, n_bins=8):
    """Compute RMSE/MSE values grouped by true-temperature bins."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Stable bins even when temperatures are nearly constant.
    t_min = float(np.min(y_true))
    t_max = float(np.max(y_true))
    if abs(t_max - t_min) < 1e-8:
        centers = np.array([t_min], dtype=np.float64)
        mse = np.array([float(np.mean((y_pred - y_true) ** 2))], dtype=np.float64)
        rmse = np.sqrt(mse)
        return centers, mse, rmse

    edges = np.linspace(t_min, t_max, n_bins + 1)
    centers = []
    mse_vals = []
    rmse_vals = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (y_true >= lo) & (y_true < hi)
        else:
            mask = (y_true >= lo) & (y_true <= hi)
        if not np.any(mask):
            continue
        err = y_pred[mask] - y_true[mask]
        mse_i = float(np.mean(err ** 2))
        centers.append((lo + hi) * 0.5)
        mse_vals.append(mse_i)
        rmse_vals.append(float(np.sqrt(mse_i)))

    return np.asarray(centers), np.asarray(mse_vals), np.asarray(rmse_vals)


def plot_experiment_results(
    history,
    y_true,
    y_pred,
    output_dir="results",
    y_true_series=None,
    y_pred_series=None,
    metric_unit="normalized",
):
    """Save training curves and prediction comparison charts."""
    os.makedirs(output_dir, exist_ok=True)
    paper_x = None
    paper_true = None
    paper_pred = None
    paper_temp_centers = None
    paper_rmse = None
    paper_mse = None

    # 1) Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=180)
    plt.close()

    # 2) Physical params and LR curves
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["epoch"], history["r_in"], label="R_in", color="tab:blue", linewidth=2)
    ax1.plot(history["epoch"], history["h_conv"], label="h_conv", color="tab:orange", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Physical Parameters")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["lr"], label="LR", color="tab:green", linestyle="--", linewidth=2)
    ax2.set_ylabel("Learning Rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title("Learned Physical Parameters and Learning Rate")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "params_and_lr.png"), dpi=180)
    plt.close(fig)

    # 3) Prediction vs ground truth
    if len(y_true) > 0 and len(y_pred) > 0:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sample_count = min(3000, len(y_true))
        idx = np.random.choice(len(y_true), sample_count, replace=False)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true[idx], y_pred[idx], s=8, alpha=0.45, color="tab:purple")
        min_v = float(min(y_true[idx].min(), y_pred[idx].min()))
        max_v = float(max(y_true[idx].max(), y_pred[idx].max()))
        plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5, label="Ideal")
        plt.xlabel("True Temperature (normalized)")
        plt.ylabel("Predicted Temperature (normalized)")
        plt.title("Prediction vs Ground Truth")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "prediction_scatter.png"), dpi=180)
        plt.close()

    # 4) Predicted vs True values (line chart across sample index/time steps)
    if y_true_series is None or y_pred_series is None:
        if len(y_true) > 0 and len(y_pred) > 0:
            n_series = min(200, len(y_true))
            y_true_series = np.asarray(y_true)[:n_series]
            y_pred_series = np.asarray(y_pred)[:n_series]

    if y_true_series is not None and y_pred_series is not None and len(y_true_series) > 0:
        y_true_series = np.asarray(y_true_series).reshape(-1)
        y_pred_series = np.asarray(y_pred_series).reshape(-1)
        x_axis = np.arange(len(y_true_series))
        paper_x = x_axis
        paper_true = y_true_series
        paper_pred = y_pred_series
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, y_pred_series, color="tab:blue", linewidth=2.0, label="Predicted")
        plt.plot(x_axis, y_true_series, color="tab:red", linestyle="--", linewidth=2.0, label="True")
        plt.xlabel("Time steps / sample index")
        plt.ylabel(f"Temperature ({metric_unit})")
        plt.title("Predicted vs True values")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predicted_vs_true_values.png"), dpi=180)
        plt.close()

    # 5) RMSE/MSE vs Temperature (binned by true temperature)
    if len(y_true) > 0 and len(y_pred) > 0:
        centers, mse_vals, rmse_vals = _compute_binned_error_curves(y_true, y_pred, n_bins=8)
        if len(centers) > 0:
            paper_temp_centers = centers
            paper_rmse = rmse_vals
            paper_mse = mse_vals
            # Combined panel
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            ax1.plot(centers, rmse_vals, marker="o", linewidth=2, color="tab:blue")
            ax1.set_xlabel(f"Temperature ({metric_unit})")
            ax1.set_ylabel(f"RMSE ({metric_unit})")
            ax1.set_title("RMSE vs Temperature")
            ax1.grid(alpha=0.3)

            ax2.plot(centers, mse_vals, marker="o", linewidth=2, color="tab:green")
            ax2.set_xlabel(f"Temperature ({metric_unit})")
            ax2.set_ylabel(f"MSE ({metric_unit}^2)")
            ax2.set_title("MSE vs Temperature")
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "error_vs_temperature.png"), dpi=180)
            plt.close(fig)

            # Separate files to match common paper plotting style.
            plt.figure(figsize=(6.5, 5))
            plt.plot(centers, rmse_vals, marker="o", linewidth=2, color="tab:blue")
            plt.xlabel(f"Temperature ({metric_unit})")
            plt.ylabel(f"RMSE ({metric_unit})")
            plt.title("RMSE vs Temperature")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "rmse_vs_temperature.png"), dpi=180)
            plt.close()

            plt.figure(figsize=(6.5, 5))
            plt.plot(centers, mse_vals, marker="o", linewidth=2, color="tab:green")
            plt.xlabel(f"Temperature ({metric_unit})")
            plt.ylabel(f"MSE ({metric_unit}^2)")
            plt.title("MSE vs Temperature")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "mse_vs_temperature.png"), dpi=180)
            plt.close()

    # 6) Paper-style summary panel (1x3)
    if (
        paper_x is not None
        and paper_temp_centers is not None
        and paper_rmse is not None
        and paper_mse is not None
    ):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

        axes[0].plot(paper_x, paper_pred, color="tab:blue", linewidth=2.0, label="Predicted")
        axes[0].plot(paper_x, paper_true, color="tab:red", linestyle="--", linewidth=2.0, label="True")
        axes[0].set_title("Predicted vs True values")
        axes[0].set_xlabel("Time steps / sample index")
        axes[0].set_ylabel(f"Temperature ({metric_unit})")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(paper_temp_centers, paper_rmse, marker="o", color="tab:blue", linewidth=2.0)
        axes[1].set_title("RMSE vs Temperature")
        axes[1].set_xlabel(f"Temperature ({metric_unit})")
        axes[1].set_ylabel(f"RMSE ({metric_unit})")
        axes[1].grid(alpha=0.25)

        axes[2].plot(paper_temp_centers, paper_mse, marker="o", color="tab:green", linewidth=2.0)
        axes[2].set_title("MSE vs Temperature")
        axes[2].set_xlabel(f"Temperature ({metric_unit})")
        axes[2].set_ylabel(f"MSE ({metric_unit}^2)")
        axes[2].grid(alpha=0.25)

        fig.suptitle("Model Performance Overview", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "paper_style_summary.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, mode="temporal", seed=42):
    """Split dataset either temporally (recommended) or randomly."""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    if mode == "random":
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
    else:
        train_dataset = Subset(dataset, range(0, train_size))
        val_dataset = Subset(dataset, range(train_size, train_size + val_size))
        test_dataset = Subset(dataset, range(train_size + val_size, total_size))

    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size


def build_fixed_subset(dataset, max_samples, seed=42):
    """Build a deterministic subset for stable validation/test metrics."""
    size = len(dataset)
    if max_samples is None or max_samples <= 0 or max_samples >= size:
        return dataset, size

    generator = torch.Generator().manual_seed(seed)
    fixed_indices = torch.randperm(size, generator=generator)[:max_samples].tolist()
    return Subset(dataset, fixed_indices), len(fixed_indices)


def infer_temperature_scale(dataset):
    """Return (y_min, y_max) if a consistent scale exists, otherwise None."""
    if isinstance(dataset, BatteryPINNDataset):
        if hasattr(dataset, "y_min") and hasattr(dataset, "y_max"):
            return float(dataset.y_min), float(dataset.y_max)
        return None

    if isinstance(dataset, Subset):
        return infer_temperature_scale(dataset.dataset)

    if isinstance(dataset, ConcatDataset):
        scales = []
        for ds in dataset.datasets:
            scale = infer_temperature_scale(ds)
            if scale is None:
                return None
            scales.append(scale)

        # If all datasets use the same min-max range, we can safely invert.
        first = scales[0]
        if all(abs(s[0] - first[0]) < 1e-6 and abs(s[1] - first[1]) < 1e-6 for s in scales[1:]):
            return first
        return None

    return None

def run_project(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_global_seed(args.seed, deterministic=args.deterministic)

    if device.type == "cuda":
        if not args.deterministic:
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Unified output folders for all artifacts.
    output_root = args.output_dir
    figures_dir = os.path.join(output_root, "figures")
    checkpoint_path = os.path.join(output_root, "best_model.pth")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. 准备数据
    if args.csv_files:
        csv_files = [item.strip() for item in args.csv_files.split(",") if item.strip()]
    else:
        csv_files = [args.csv_file]

    # Auto-resolve relative filenames into ./data for convenience.
    resolved_csv_files = []
    for p in csv_files:
        if os.path.exists(p):
            resolved_csv_files.append(p)
            continue
        candidate = os.path.join("data", p)
        resolved_csv_files.append(candidate if os.path.exists(candidate) else p)
    csv_files = resolved_csv_files

    # -------- 训练速度/稳定性配置 --------
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    train_samples_per_epoch = args.train_samples_per_epoch
    val_samples = args.val_samples
    test_samples = args.test_samples
    pde_every_n_batches = args.pde_every_n_batches
    pde_warmup_epochs = args.pde_warmup_epochs
    pde_subset_size = args.pde_subset_size
    max_pde_weight = args.max_pde_weight
    pde_ramp_epochs = args.pde_ramp_epochs
    use_adaptive_pde_weight = args.use_adaptive_pde_weight
    current_pde_every_n_batches = pde_every_n_batches
    current_max_pde_weight = max_pde_weight
    num_workers = args.num_workers if args.num_workers >= 0 else min(8, os.cpu_count() or 4)
    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    
    # 检查CSV文件是否存在
    missing_files = [p for p in csv_files if not os.path.exists(p)]
    if missing_files:
        print("❌ 错误：以下文件不存在")
        for p in missing_files:
            print(f"   - {p}")
        print(f"   当前工作目录: {os.getcwd()}")
        return

    print("Training with CSV files:")
    for p in csv_files:
        print(f"  - {p}")

    # 加载完整数据（支持多文件拼接）
    dataset = build_dataset_from_csv_files(
        csv_files,
        global_y_normalize=args.global_y_normalize_multi,
    )
    
    # 2. 数据集分割 (70/20/10 用于训练/验证/测试)
    train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.2,
        mode=args.split_mode,
        seed=args.seed,
    )
    
    # 使用采样器缩短每轮时长，同时覆盖大规模数据
    # Train sampling: avoid aggressive oversampling by default to reduce overfitting.
    requested_train_samples = max(1, int(train_samples_per_epoch))
    if args.train_sampler_replacement:
        train_num_samples = requested_train_samples
        train_sampler = torch.utils.data.RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=train_num_samples,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            drop_last=True,
            **loader_kwargs,
        )
    else:
        train_num_samples = min(train_size, requested_train_samples)
        if train_num_samples < train_size:
            train_subset, _ = build_fixed_subset(train_dataset, train_num_samples, seed=args.seed)
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=True,
                **loader_kwargs,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                drop_last=True,
                **loader_kwargs,
            )

    # Validation/test sampling: fixed subset (or full set) for reproducible curves.
    val_eval_dataset, val_num_samples = build_fixed_subset(val_dataset, val_samples, seed=args.seed + 7)
    test_eval_dataset, test_num_samples = build_fixed_subset(test_dataset, test_samples, seed=args.seed + 13)

    val_loader = torch.utils.data.DataLoader(
        val_eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    
    print(f"Data split ({args.split_mode}) - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print(
        f"Sampling per epoch - Train: {train_num_samples}, Val: {val_num_samples}, Test: {test_num_samples}"
    )
    print(
        f"Sampling strategy - train_replacement={args.train_sampler_replacement}, "
        f"val_fixed_subset=True, test_fixed_subset=True"
    )
    print(
        f"Loader config - train_batch={train_batch_size}, eval_batch={eval_batch_size}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )

    # 3. 初始化模型与优化器
    model = MM_Q3D_KAN_PINN(
        use_am_attention=args.use_am_attention,
        am_dropout=args.am_dropout,
    ).to(device)
    
    # 优化器升级为 AdamW，配合余弦退火效果更好
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 引入余弦退火学习率调度器
    # T_max 设为 args.epochs (总训练轮数)，eta_min 设为 1e-6 (极小的保底学习率)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    criterion = nn.SmoothL1Loss(beta=0.02)
    # Keep AMP scaler compatible with both newer and older PyTorch APIs.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # 4. 训练循环
    print("\nStarting MM-Q3D-KAN-PINN Training...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    best_epoch = 0
    last_epoch = 0
    pde_backoff_events = 0
    pde_bad_val_streak = 0
    stage1_epochs = max(0, args.stage1_epochs)
    if stage1_epochs > 0:
        print(f"Two-stage training enabled: stage1(data-only)={stage1_epochs} epochs, then PINN stage2")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "r_in": [],
        "h_conv": [],
    }
    
    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        use_pde_this_epoch = args.use_pde and epoch > stage1_epochs

        # 训练
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_pde=use_pde_this_epoch,
            log_every=args.log_every,
            epoch=epoch,
            pde_every_n_batches=current_pde_every_n_batches,
            pde_warmup_epochs=pde_warmup_epochs,
            pde_subset_size=pde_subset_size,
            max_pde_weight=current_max_pde_weight,
            pde_ramp_epochs=pde_ramp_epochs,
            use_adaptive_pde_weight=use_adaptive_pde_weight,
            amp_enabled=True,
            scaler=scaler,
        )
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        # 打印物理参数辨识结果
        if device.type == "cuda":
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            mem_text = f" | GPU Mem(MiB): alloc={mem_alloc:.1f}, reserv={mem_reserved:.1f}, peak={mem_peak:.1f}"
        else:
            mem_text = ""

        r_in_param = getattr(model, "R_in", None)
        if r_in_param is None:
            r_in_param = getattr(model, "R_in_base")

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"R_in: {r_in_param.item():.4f} | h: {model.h_conv.item():.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | PINN: {use_pde_this_epoch} "
              f"| pde_w={current_max_pde_weight:.4f} | pde_every={current_pde_every_n_batches}{mem_text}")
        
        # 学习率调度 (余弦退火只跟当前 epoch 有关，不需要传 val_loss)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["r_in"].append(float(r_in_param.item()))
        history["h_conv"].append(float(model.h_conv.item()))

        if use_pde_this_epoch and args.auto_pde_backoff and best_val_loss < float("inf"):
            worsen_ratio = (val_loss - best_val_loss) / max(best_val_loss, 1e-12)
            if worsen_ratio > args.pde_backoff_rel_increase:
                pde_bad_val_streak += 1
            else:
                pde_bad_val_streak = 0

            if pde_bad_val_streak >= args.pde_backoff_patience:
                new_max_pde_weight = max(args.min_pde_weight, current_max_pde_weight * args.pde_backoff_factor)
                new_pde_every = min(
                    args.max_pde_every_n_batches,
                    max(1, int(round(current_pde_every_n_batches * args.pde_every_backoff_factor))),
                )
                if new_max_pde_weight < current_max_pde_weight or new_pde_every > current_pde_every_n_batches:
                    pde_backoff_events += 1
                    print(
                        "⚙️  Auto PDE backoff triggered: "
                        f"max_pde_weight {current_max_pde_weight:.4f}->{new_max_pde_weight:.4f}, "
                        f"pde_every_n_batches {current_pde_every_n_batches}->{new_pde_every}"
                    )
                current_max_pde_weight = new_max_pde_weight
                current_pde_every_n_batches = new_pde_every
                pde_bad_val_streak = 0
        
        # 早停检查和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"✓ Saved best model at epoch {epoch}: {checkpoint_path}")
        else:
            patience_counter += 1
            if epoch >= args.min_epochs_before_early_stop and patience_counter >= patience:
                print(f"\n⊘ Early stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break
    
    print("=" * 70)
    
    # 5. 加载最佳模型进行测试
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch']}: {checkpoint_path}")
    
    # 6. 测试集评估
    test_loss = validate(model, test_loader, criterion, device)
    print(f"\n📊 Test Loss: {test_loss:.6f}")

    # 7. 收集测试预测用于画图
    pred_buffer = []
    true_buffer = []
    vis_pred_buffer = []
    vis_true_buffer = []
    model.eval()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred_y = model(x_batch)
            pred_buffer.append(pred_y.detach().cpu().numpy().reshape(-1))
            true_buffer.append(y_batch.detach().cpu().numpy().reshape(-1))
            if i >= 49:  # use up to 50 test batches for visualization
                break

    # Build an ordered series view for Predicted-vs-True line chart.
    vis_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        taken = 0
        for x_batch, y_batch in vis_loader:
            x_batch = x_batch.to(device)
            pred_y = model(x_batch).detach().cpu().numpy().reshape(-1)
            true_y = y_batch.detach().cpu().numpy().reshape(-1)
            remaining = max(0, args.plot_series_points - taken)
            if remaining <= 0:
                break
            vis_pred_buffer.append(pred_y[:remaining])
            vis_true_buffer.append(true_y[:remaining])
            taken += min(len(pred_y), remaining)
            if taken >= args.plot_series_points:
                break

    y_pred_for_plot = np.concatenate(pred_buffer) if pred_buffer else np.array([])
    y_true_for_plot = np.concatenate(true_buffer) if true_buffer else np.array([])
    y_pred_series = np.concatenate(vis_pred_buffer) if vis_pred_buffer else np.array([])
    y_true_series = np.concatenate(vis_true_buffer) if vis_true_buffer else np.array([])

    temp_scale = infer_temperature_scale(dataset)
    test_mae_c = None
    test_rmse_c = None
    test_mse_c = None
    metric_unit = "normalized"
    y_pred_metric = y_pred_for_plot
    y_true_metric = y_true_for_plot
    y_pred_series_metric = y_pred_series
    y_true_series_metric = y_true_series
    if temp_scale is not None and len(y_pred_for_plot) > 0:
        y_min, y_max = temp_scale
        y_span = max(1e-8, (y_max - y_min))
        y_pred_c = y_pred_for_plot * y_span + y_min
        y_true_c = y_true_for_plot * y_span + y_min
        y_pred_series_c = y_pred_series * y_span + y_min if len(y_pred_series) > 0 else y_pred_series
        y_true_series_c = y_true_series * y_span + y_min if len(y_true_series) > 0 else y_true_series
        test_mse_c = float(np.mean((y_pred_c - y_true_c) ** 2))
        test_mae_c = float(np.mean(np.abs(y_pred_c - y_true_c)))
        test_rmse_c = float(np.sqrt(np.mean((y_pred_c - y_true_c) ** 2)))
        metric_unit = "°C"
        y_pred_metric = y_pred_c
        y_true_metric = y_true_c
        y_pred_series_metric = y_pred_series_c
        y_true_series_metric = y_true_series_c
        print(f"🌡️ Test MAE: {test_mae_c:.4f} °C | Test RMSE: {test_rmse_c:.4f} °C | Test MSE: {test_mse_c:.4f} °C^2")
    elif isinstance(dataset, ConcatDataset):
        print("ℹ️ Skipped °C metrics: merged datasets use different normalization scales.")

    # 8. 绘图并保存
    plot_experiment_results(
        history,
        y_true_metric,
        y_pred_metric,
        output_dir=figures_dir,
        y_true_series=y_true_series_metric,
        y_pred_series=y_pred_series_metric,
        metric_unit=metric_unit,
    )
    print(
        f"📈 Saved charts in {figures_dir}: "
        "loss_curves.png, params_and_lr.png, prediction_scatter.png, "
        "predicted_vs_true_values.png, rmse_vs_temperature.png, mse_vs_temperature.png, "
        "paper_style_summary.png"
    )

    # 9. 保存实验摘要，便于对比不同配置结果
    summary_path = os.path.join(output_root, "experiment_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("MM-Q3D-KAN-PINN Experiment Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Device: {device}\n")
        f.write(f"CSV Files: {', '.join(csv_files)}\n")
        f.write(f"Global Y normalize (multi): {args.global_y_normalize_multi}\n")
        f.write(f"Split mode: {args.split_mode}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Stage1 epochs (data-only): {stage1_epochs}\n")
        f.write(f"Epochs completed: {last_epoch}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        if test_mae_c is not None and test_rmse_c is not None:
            f.write(f"Test MAE (°C): {test_mae_c:.6f}\n")
            f.write(f"Test RMSE (°C): {test_rmse_c:.6f}\n")
        if test_mse_c is not None:
            f.write(f"Test MSE (°C^2): {test_mse_c:.6f}\n")
        f.write(f"Train Samples/Epoch: {train_num_samples}\n")
        f.write(f"Val Samples: {val_num_samples}\n")
        f.write(f"Test Samples: {test_num_samples}\n")
        f.write(f"Train Batch: {train_batch_size}\n")
        f.write(f"Eval Batch: {eval_batch_size}\n")
        f.write(f"PDE every N batches (init): {pde_every_n_batches}\n")
        f.write(f"PDE every N batches (final): {current_pde_every_n_batches}\n")
        f.write(f"PDE warmup epochs: {pde_warmup_epochs}\n")
        f.write(f"PDE subset size: {pde_subset_size}\n")
        f.write(f"Max PDE weight (init): {max_pde_weight}\n")
        f.write(f"Max PDE weight (final): {current_max_pde_weight}\n")
        f.write(f"PDE ramp epochs: {pde_ramp_epochs}\n")
        f.write(f"Adaptive PDE weight: {use_adaptive_pde_weight}\n")
        f.write(f"Auto PDE backoff: {args.auto_pde_backoff}\n")
        f.write(f"Auto PDE backoff events: {pde_backoff_events}\n")
        f.write(f"Deterministic mode: {args.deterministic}\n")
        f.write(f"Use AM adaptive self-attention: {args.use_am_attention}\n")
        f.write(f"AM dropout: {args.am_dropout}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
    print(f"🧾 Saved summary: {summary_path}")

    # 10. 简单验证 (取一些样本看输出)
    model.eval()
    print("\nSample Predictions (first 5 samples):")
    print("-" * 50)
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            if i >= 1:  # 只取一个batch
                break
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred_y = model(x_batch)
            
            for j in range(min(5, len(x_batch))):
                pred_val = pred_y[j].item()
                true_val = y_batch[j].item()
                error = abs(pred_val - true_val)
                print(f"Sample {j+1}: Pred={pred_val:.4f} | True={true_val:.4f} | Error={error:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MM-Q3D-KAN-PINN training entry")
    parser.add_argument("--csv-file", type=str, default="data/TR_1_2109.csv")
    parser.add_argument(
        "--csv-files",
        type=str,
        default="",
        help="Comma-separated CSV list for merged training, e.g. TR_1_2109.csv,TR_2_0111.csv",
    )
    parser.add_argument("--global-y-normalize-multi", action="store_true", default=True)
    parser.add_argument("--no-global-y-normalize-multi", dest="global_y_normalize_multi", action="store_false")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=40)
    parser.add_argument("--split-mode", type=str, choices=["temporal", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--train-samples-per-epoch", type=int, default=300_000)
    parser.add_argument("--val-samples", type=int, default=0, help="0 means evaluate on full validation set")
    parser.add_argument("--test-samples", type=int, default=0, help="0 means evaluate on full test set")
    parser.add_argument("--train-sampler-replacement", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=-1, help="-1 means auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--use-pde", action="store_true", default=True)
    parser.add_argument("--no-pde", dest="use_pde", action="store_false")
    parser.add_argument("--pde-every-n-batches", type=int, default=20)
    parser.add_argument("--pde-warmup-epochs", type=int, default=0)
    parser.add_argument("--pde-subset-size", type=int, default=128)
    parser.add_argument("--max-pde-weight", type=float, default=0.05)
    parser.add_argument("--pde-ramp-epochs", type=int, default=30)
    parser.add_argument("--auto-pde-backoff", action="store_true", default=True)
    parser.add_argument("--no-auto-pde-backoff", dest="auto_pde_backoff", action="store_false")
    parser.add_argument("--pde-backoff-patience", type=int, default=3)
    parser.add_argument("--pde-backoff-rel-increase", type=float, default=0.02)
    parser.add_argument("--pde-backoff-factor", type=float, default=0.7)
    parser.add_argument("--min-pde-weight", type=float, default=0.005)
    parser.add_argument("--pde-every-backoff-factor", type=float, default=1.5)
    parser.add_argument("--max-pde-every-n-batches", type=int, default=100)
    parser.add_argument("--use-adaptive-pde-weight", action="store_true", default=True)
    parser.add_argument("--no-adaptive-pde-weight", dest="use_adaptive_pde_weight", action="store_false")
    parser.add_argument("--use-am-attention", action="store_true", default=True)
    parser.add_argument("--no-am-attention", dest="use_am_attention", action="store_false")
    parser.add_argument("--am-dropout", type=float, default=0.05)
    parser.add_argument("--plot-series-points", type=int, default=120)
    run_project(parser.parse_args())