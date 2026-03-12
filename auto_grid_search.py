import argparse
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime


def parse_list_int(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_list_float(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def read_summary(summary_path):
    result = {
        "best_val_loss": None,
        "test_loss": None,
        "test_mae_degC": None,
        "test_rmse_degC": None,
        "test_mse_degC2": None,
        "epochs_completed": None,
        "best_epoch": None,
    }

    if not os.path.exists(summary_path):
        return result

    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Best Val Loss:"):
                result["best_val_loss"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Test Loss:"):
                result["test_loss"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Test MAE (degC):") or line.startswith("Test MAE (°C):"):
                result["test_mae_degC"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Test RMSE (degC):") or line.startswith("Test RMSE (°C):"):
                result["test_rmse_degC"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Test MSE (degC^2):") or line.startswith("Test MSE (°C^2):"):
                result["test_mse_degC2"] = float(line.split(":", 1)[1].strip())
            elif line.startswith("Epochs completed:"):
                result["epochs_completed"] = int(float(line.split(":", 1)[1].strip()))
            elif line.startswith("Best epoch:"):
                result["best_epoch"] = int(float(line.split(":", 1)[1].strip()))

    return result


def run_one(base_cmd, run_name):
    cmd = base_cmd + ["--output-dir", run_name]
    print("\n" + "=" * 90)
    print("Running:", " ".join(cmd))
    print("=" * 90)
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        print(f"Run failed: {run_name}, return code={completed.returncode}")
    return completed.returncode


def fmt(v, digits=6):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def metric_value(result, rank_by):
    v = result.get(rank_by)
    return v if v is not None else 1e18


def main():
    parser = argparse.ArgumentParser(description="Auto grid search for MM-Q3D-KAN-PINN")
    parser.add_argument("--csv-files", type=str, default="data/TR_1_2109.csv,data/TR_2_0111.csv,data/TR_3_2811.csv")
    parser.add_argument("--base-output-root", type=str, default="outputs_grid")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=40)
    parser.add_argument("--split-mode", type=str, choices=["temporal", "random"], default="temporal")
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--train-samples-per-epoch", type=int, default=300000)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-grid", type=str, default="")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pde-subset-size", type=int, default=128)
    parser.add_argument("--pde-ramp-epochs", type=int, default=30)
    parser.add_argument("--use-am-attention", action="store_true", default=True)
    parser.add_argument("--no-am-attention", dest="use_am_attention", action="store_false")
    parser.add_argument("--am-dropout", type=float, default=0.05)
    parser.add_argument("--stage1-grid", type=str, default="20,30")
    parser.add_argument("--max-pde-weight-grid", type=str, default="0.03,0.05")
    parser.add_argument("--pde-every-n-batches-grid", type=str, default="20")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["test_loss", "test_rmse_degC", "test_mae_degC", "best_val_loss"],
        default="test_rmse_degC",
    )
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument(
        "--resume-root",
        type=str,
        default="",
        help="If set, append/refresh report for an existing grid root instead of creating a new timestamp folder.",
    )
    args = parser.parse_args()

    stage1_grid = parse_list_int(args.stage1_grid)
    pde_every_grid = parse_list_int(args.pde_every_n_batches_grid)
    max_pde_weight_grid = parse_list_float(args.max_pde_weight_grid)
    lr_grid = parse_list_float(args.lr_grid) if args.lr_grid.strip() else [args.lr]

    if not stage1_grid or not pde_every_grid or not max_pde_weight_grid or not lr_grid:
        raise ValueError("Grid lists must not be empty")

    if args.resume_root:
        root = args.resume_root
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(args.base_output_root, ts)
    os.makedirs(root, exist_ok=True)

    combos = list(itertools.product(stage1_grid, max_pde_weight_grid, pde_every_grid, lr_grid))
    print(f"Total runs: {len(combos)}")
    print(f"Output root: {root}")

    results = []
    for idx, (stage1_epochs, max_pde_weight, pde_every_n_batches, lr) in enumerate(combos, start=1):
        run_name = os.path.join(
            root,
            f"run_{idx:02d}_s1_{stage1_epochs}_wp_{max_pde_weight:.3f}_pe_{pde_every_n_batches}_lr_{lr:.1e}",
        )
        os.makedirs(run_name, exist_ok=True)
        summary_path = os.path.join(run_name, "experiment_summary.txt")

        cmd = [
            sys.executable,
            "main.py",
            "--csv-files",
            args.csv_files,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--min-epochs-before-early-stop",
            str(args.min_epochs_before_early_stop),
            "--split-mode",
            args.split_mode,
            "--train-batch-size",
            str(args.train_batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--train-samples-per-epoch",
            str(args.train_samples_per_epoch),
            "--val-samples",
            str(args.val_samples),
            "--test-samples",
            str(args.test_samples),
            "--lr",
            str(lr),
            "--weight-decay",
            str(args.weight_decay),
            "--stage1-epochs",
            str(stage1_epochs),
            "--pde-every-n-batches",
            str(pde_every_n_batches),
            "--pde-subset-size",
            str(args.pde_subset_size),
            "--max-pde-weight",
            str(max_pde_weight),
            "--pde-ramp-epochs",
            str(args.pde_ramp_epochs),
            "--am-dropout",
            str(args.am_dropout),
            "--log-every",
            str(args.log_every),
        ]
        if not args.use_am_attention:
            cmd.append("--no-am-attention")

        if args.skip_existing and os.path.exists(summary_path):
            print(f"Skipping existing run: {run_name}")
            ret = 0
        else:
            ret = run_one(cmd, run_name)

        summary = read_summary(summary_path)
        result = {
            "run_name": run_name,
            "return_code": ret,
            "stage1_epochs": stage1_epochs,
            "max_pde_weight": max_pde_weight,
            "pde_every_n_batches": pde_every_n_batches,
            "lr": lr,
            **summary,
        }
        results.append(result)

    # Rank by chosen metric (fallback to large number when missing)
    results_sorted = sorted(
        results,
        key=lambda x: metric_value(x, args.rank_by),
    )

    report_path = os.path.join(root, "grid_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MM-Q3D-KAN-PINN Grid Search Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total runs: {len(results_sorted)}\n")
        f.write(f"CSV files: {args.csv_files}\n")
        f.write(f"Ranking metric: {args.rank_by}\n")
        f.write(f"Skip existing: {args.skip_existing}\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        f.write(
            "Rank | TestLoss | ValBest | MAE(°C) | RMSE(°C) | MSE(°C^2) | s1 | w_pde | pde_every | lr | "
            "epochs_done | best_epoch | return | run\n"
        )
        f.write("-" * 80 + "\n")

        for i, r in enumerate(results_sorted, start=1):
            f.write(
                f"{i:>4} | "
                f"{fmt(r['test_loss']):>8} | "
                f"{fmt(r['best_val_loss']):>7} | "
                f"{fmt(r['test_mae_degC']):>9} | "
                f"{fmt(r['test_rmse_degC']):>10} | "
                f"{fmt(r['test_mse_degC2']):>12} | "
                f"{r['stage1_epochs']:>2} | "
                f"{r['max_pde_weight']:>5} | "
                f"{r['pde_every_n_batches']:>9} | "
                f"{r['lr']:.1e} | "
                f"{fmt(r['epochs_completed'], 0):>11} | "
                f"{fmt(r['best_epoch'], 0):>9} | "
                f"{r['return_code']:>6} | "
                f"{r['run_name']}\n"
            )

    csv_path = os.path.join(root, "grid_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "run_name",
                "return_code",
                "stage1_epochs",
                "max_pde_weight",
                "pde_every_n_batches",
                "lr",
                "best_val_loss",
                "test_loss",
                "test_mae_degC",
                "test_rmse_degC",
                "test_mse_degC2",
                "epochs_completed",
                "best_epoch",
            ],
        )
        writer.writeheader()
        for rank, r in enumerate(results_sorted, start=1):
            row = {"rank": rank, **r}
            writer.writerow(row)

    print("\nGrid search completed.")
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    if results_sorted:
        best = results_sorted[0]
        print("Best run:")
        print(best)
        best_cmd = (
            f"python main.py --csv-files {args.csv_files} --split-mode {args.split_mode} "
            f"--epochs {args.epochs} --patience {args.patience} "
            f"--min-epochs-before-early-stop {args.min_epochs_before_early_stop} "
            f"--train-batch-size {args.train_batch_size} --eval-batch-size {args.eval_batch_size} "
            f"--train-samples-per-epoch {args.train_samples_per_epoch} --val-samples {args.val_samples} "
            f"--test-samples {args.test_samples} --weight-decay {args.weight_decay} "
            f"--stage1-epochs {best['stage1_epochs']} --pde-every-n-batches {best['pde_every_n_batches']} "
            f"--pde-subset-size {args.pde_subset_size} --max-pde-weight {best['max_pde_weight']} "
            f"--pde-ramp-epochs {args.pde_ramp_epochs} --lr {best['lr']} "
            f"--am-dropout {args.am_dropout} --output-dir <your_output_dir>"
        )
        if not args.use_am_attention:
            best_cmd += " --no-am-attention"
        print("Suggested best-command template:")
        print(best_cmd)


if __name__ == "__main__":
    main()
