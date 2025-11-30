"""
Quantization experiments for NAFNet-small on SIDD/BSDS.

Two experiments are covered:
1) Dynamic quantization (INT8 weight-only and FP16 weight-only) using PyTorch's
   built-in dynamic quantization utilities (no calibration).
2) Static post-training quantization (PTQ INT8) with calibration that targets
   the conv-heavy architecture via QuantStub/DeQuantStub wrapping.

For each run the script:
- loads the FP32 checkpoint,
- applies the requested quantization recipe,
- saves a labeled `.pth` checkpoint,
- evaluates on a small subset to report PSNR/SSIM and latency,
- prints a comparison table against the FP32 baseline.

Usage example:
    python -m src.quantize.quantize_nafnet --dataset sidd --checkpoint src/checkpoints/nafnet_small_best_sidd.pth
"""

import argparse
import copy
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.ao.quantization as tq
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.models.nafnet import NAFNet

# Default locations
DEFAULT_CHECKPOINT = "src/checkpoints/nafnet_small_best_sidd.pth"
DEFAULT_SAVE_DIR = "src/checkpoints/quantized"
ARCH_NAME = "NAFNet-small"

# Quantization configurations to try. Feel free to extend.
QUANT_CONFIGS: List[Dict] = [
    {
        "name": "fp32_baseline",
        "precision": "FP32",
        "method": "none",
        "description": "Reload FP32 checkpoint without quantization.",
    },
    {
        "name": "dynamic_int8",
        "precision": "INT8-dynamic",
        "method": "dynamic",
        "modules": [torch.nn.Conv2d, torch.nn.Linear],
        "dtype": torch.qint8,
        "description": "Dynamic weight-only quantization to INT8 (no calibration).",
    },
    {
        "name": "dynamic_fp16",
        "precision": "FP16-dynamic",
        "method": "dynamic",
        "modules": [torch.nn.Conv2d, torch.nn.Linear],
        "dtype": torch.float16,
        "description": "Dynamic weight-only casting to float16 (no calibration).",
    },
    {
        "name": "static_ptq_int8",
        "precision": "INT8-PTQ",
        "method": "static",
        "description": "Static PTQ INT8 with calibration focusing on conv layers.",
    },
]


class QuantWrapper(nn.Module):
    """
    Lightweight wrapper that inserts quant/dequant stubs so we can quantize
    the conv-heavy NAFNet without editing the model definition.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.dequant(self.model(self.quant(x)))


def load_fp32_model(checkpoint_path: str, device: torch.device) -> NAFNet:
    """Load the FP32 NAFNet-small model from checkpoint."""
    model = NAFNet(
        image_channels=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 8],
        dec_blk_nums=[1, 1, 1, 1],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def apply_dynamic_quantization(
    model: nn.Module,
    dtype: torch.dtype,
    modules: Sequence[type],
) -> nn.Module:
    """Apply dynamic quantization or weight-only FP16 casting."""
    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()
    module_set = set(modules) if modules else {torch.nn.Linear}
    return tq.quantize_dynamic(model_cpu, module_set, dtype=dtype)


def apply_static_ptq(
    model: nn.Module,
    calib_loader: DataLoader,
    backend: str,
) -> nn.Module:
    """Static PTQ with calibration; primarily targets conv layers."""
    torch.backends.quantized.engine = backend
    wrapper = QuantWrapper(copy.deepcopy(model).to("cpu"))
    wrapper.eval()
    wrapper.qconfig = tq.get_default_qconfig(backend)

    prepared = tq.prepare(wrapper, inplace=False)
    with torch.inference_mode():
        for noisy, _ in calib_loader:
            prepared(noisy.to("cpu"))

    # IMPORTANT: Switch to 'fbgemm' for x86 CPUs or 'qnnpack' for ARM/Mobile
    # 'qnnpack' is strictly for ARM/Mobile and often fails on x86 with "NotImplementedError"
    # for certain ops like quantized::conv2d.new.
    # Since we are likely on x86 (Linux VM), we force 'fbgemm' for the conversion step.
    torch.backends.quantized.engine = 'fbgemm'
    
    converted = tq.convert(prepared, inplace=False)
    return converted


def apply_quantization(
    model: nn.Module,
    config: Dict,
    calib_loader: Optional[DataLoader],
    qengine: Optional[str],
) -> nn.Module:
    """
    Apply quantization according to the provided config.
    Supported methods: "none", "dynamic", "static".
    """
    method = config.get("method", "none")

    if method == "none":
        return model

    if method == "dynamic":
        dtype = config.get("dtype", torch.qint8)
        modules = config.get("modules", [torch.nn.Linear])
        return apply_dynamic_quantization(model, dtype=dtype, modules=modules)

    if method == "static":
        if qengine is None:
            raise RuntimeError("Static quantization requested but no quantized engine is available.")
        if calib_loader is None:
            raise RuntimeError("Static quantization requested but no calibration loader was provided.")
        return apply_static_ptq(model, calib_loader=calib_loader, backend=qengine)

    raise ValueError(f"Unknown quantization method '{method}'")


def count_quantized_modules(model: nn.Module) -> Dict[str, int]:
    """Count quantized modules for a quick sanity check."""
    return {
        "quantized_conv": sum(isinstance(m, (nnq.Conv1d, nnq.Conv2d, nnq.Conv3d)) for m in model.modules()),
        "quantized_linear": sum(isinstance(m, nnqd.Linear) for m in model.modules()),
    }


def build_calibration_loader(dataset: str, batch_size: int, max_samples: int) -> DataLoader:
    """
    Build a small calibration loader for static PTQ.
    """
    dataset = dataset.lower()
    if dataset == "sidd":
        from src.data.sidd_dataset import SIDDDataset

        full_dataset = SIDDDataset(dirs=["SIDD_Small_sRGB_Only/Data"], size=128, random_crop=True)
    else:
        from src.data.denoise_dataset import DenoiseDataset

        full_dataset = DenoiseDataset(dirs=["BSDS300/images/test"], size=128, sigma=25 / 255.0, channels=3)

    if max_samples is not None and max_samples < len(full_dataset):
        full_dataset = Subset(full_dataset, list(range(max_samples)))

    return DataLoader(full_dataset, batch_size=batch_size, shuffle=False)


def save_checkpoint(model: nn.Module, save_path: str, config: Dict) -> float:
    """
    Save model state_dict with a minimal, serializable config summary.
    Returns: file size in MB.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    safe_config = {
        "name": config.get("name"),
        "precision": config.get("precision"),
        "method": config.get("method"),
        "description": config.get("description"),
        "dtype": str(config.get("dtype")),
        "modules": [m.__name__ if hasattr(m, "__name__") else str(m) for m in config.get("modules", [])],
    }
    torch.save({"model_state_dict": model.state_dict(), "quant_config": safe_config}, save_path)
    size_mb = os.path.getsize(save_path) / (1024**2)
    return size_mb


def evaluate_model(
    model: nn.Module,
    dataset_type: str,
    sample_count: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate PSNR/SSIM and latency on a small subset to keep the run quick.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model.to(device)
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    psnr_values: List[float] = []
    ssim_values: List[float] = []
    inference_times: List[float] = []
    cpu_runtimes: List[float] = []

    if dataset_type.lower() == "sidd":
        from src.data.sidd_dataset import SIDDDataset, split_sidd_dataset

        dataset = SIDDDataset(
            dirs=["SIDD_Small_sRGB_Only/Data"],
            size=128,
            random_crop=False,
        )
        _, _, test_set = split_sidd_dataset(dataset, seed=seed)
        if len(test_set) == 0:
            raise ValueError("SIDD test split is empty; cannot evaluate.")
        sample_indices = random.sample(range(len(test_set)), min(sample_count, len(test_set)))

        for idx in sample_indices:
            cpu_start_time = time.time()
            noisy, clean = test_set[idx]
            noisy = noisy.unsqueeze(0).to(device)
            clean = clean.unsqueeze(0).to(device)

            inference_start = time.time()
            with torch.inference_mode():
                denoised = model(noisy)
            inference_time = time.time() - inference_start

            psnr_value = psnr_metric(denoised, clean)
            ssim_value = ssim_metric(denoised, clean)

            cpu_runtime = time.time() - cpu_start_time

            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            inference_times.append(inference_time)
            cpu_runtimes.append(cpu_runtime)
    else:
        from src.data.denoise_dataset import DenoiseDataset

        dataset = DenoiseDataset(dirs=["BSDS300/images/test"], size=128, sigma=25 / 255.0, channels=3)
        if len(dataset) == 0:
            raise ValueError("BSDS300 test set is empty; cannot evaluate.")
        sample_indices = random.sample(range(len(dataset)), min(sample_count, len(dataset)))

        for idx in sample_indices:
            cpu_start_time = time.time()

            noisy, clean = dataset[idx]
            noisy = noisy.unsqueeze(0).to(device)
            clean = clean.unsqueeze(0).to(device)

            inference_start = time.time()
            with torch.inference_mode():
                denoised = model(noisy)
            inference_time = time.time() - inference_start

            psnr_value = psnr_metric(denoised, clean)
            ssim_value = ssim_metric(denoised, clean)

            cpu_runtime = time.time() - cpu_start_time

            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            inference_times.append(inference_time)
            cpu_runtimes.append(cpu_runtime)

    return {
        "avg_psnr": float(np.mean(psnr_values)),
        "avg_ssim": float(np.mean(ssim_values)),
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000),
        "avg_cpu_runtime_ms": float(np.mean(cpu_runtimes) * 1000),
    }


def format_table_row(columns: Sequence[str], widths: Sequence[int]) -> str:
    padded = []
    for col, width in zip(columns, widths):
        padded.append(f"{col:<{width}}")
    return " ".join(padded)


def print_results_table(results: List[Dict]):
    """Render a compact comparison table with deltas vs FP32."""
    baseline = None
    for res in results:
        if res["config"]["precision"] == "FP32":
            baseline = res
            break

    headers = [
        "Model",
        "Precision",
        "Size (MB)",
        "CPU inf (ms/img)",
        "PSNR (SIDD)",
        "SSIM (SIDD)",
        "ΔPSNR vs FP32",
        "Δinfer vs FP32",
    ]
    widths = [26, 12, 10, 17, 12, 12, 14, 16]
    print("\nRESULTS TABLE")
    print("-" * sum(widths))
    print(format_table_row(headers, widths))
    print("-" * sum(widths))

    base_psnr = baseline["metrics"]["avg_psnr"] if baseline else None
    base_infer = baseline["metrics"]["avg_inference_time_ms"] if baseline else None

    for res in results:
        metrics = res["metrics"]
        precision = res["config"]["precision"]
        model_label = f"{ARCH_NAME} {precision}"
        delta_psnr = metrics["avg_psnr"] - base_psnr if base_psnr is not None else float("nan")
        if base_infer and base_infer > 0:
            speedup = (base_infer - metrics["avg_inference_time_ms"]) / base_infer * 100
        else:
            speedup = float("nan")

        columns = [
            model_label,
            precision,
            f"{res['size_mb']:.2f}",
            f"{metrics['avg_inference_time_ms']:.2f}",
            f"{metrics['avg_psnr']:.2f}",
            f"{metrics['avg_ssim']:.4f}",
            f"{delta_psnr:+.2f}",
            f"{speedup:+.1f}%",
        ]
        print(format_table_row(columns, widths))

    print("-" * sum(widths))


def main():
    parser = argparse.ArgumentParser(description="Quantize NAFNet-small checkpoint and evaluate.")
    parser.add_argument("--dataset", choices=["sidd", "bsds"], default="sidd", help="Dataset used for evaluation.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="FP32 checkpoint path to load.")
    parser.add_argument("--save-dir", dest="save_dir", default=DEFAULT_SAVE_DIR, help="Directory to store quantized checkpoints.")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate per config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--calib-samples", type=int, default=32, help="Number of samples to use for calibration in static PTQ.")
    parser.add_argument("--calib-batch-size", type=int, default=4, help="Batch size for calibration loader.")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Determine available quantized engines in a backward-compatible way
    supported_engines: Tuple[str, ...] = ()
    if hasattr(torch.backends.quantized, "supported_engines"):
        supported_engines = tuple(torch.backends.quantized.supported_engines)
    elif hasattr(torch.backends.quantized, "supported_qengines"):
        supported_engines = tuple(torch.backends.quantized.supported_qengines)  # type: ignore[attr-defined]

    qengine: Optional[str] = None
    for candidate in ("fbgemm", "qnnpack"):
        if candidate in supported_engines:
            qengine = candidate
            break

    if qengine is not None:
        try:
            torch.backends.quantized.engine = qengine
            print(f"Using quantized engine: {qengine}")
        except Exception as err:
            print(f"Failed to set quantized engine to '{qengine}': {err}")
            qengine = None

    if qengine is None:
        print("No quantized engine available; static PTQ will be skipped.")

    calib_loader = None
    if qengine is not None and any(cfg.get("method") == "static" for cfg in QUANT_CONFIGS):
        calib_loader = build_calibration_loader(args.dataset, args.calib_batch_size, args.calib_samples)

    results: List[Dict] = []

    for config in QUANT_CONFIGS:
        method = config.get("method")
        if method == "static" and qengine is None:
            print(f"Skipping static PTQ config '{config['name']}' because no quantized engine is available.")
            continue

        print("=" * 80)
        print(f"Running config: {config['name']} | {config.get('description', '')}")

        start = time.time()
        fp32_model = load_fp32_model(args.checkpoint, device)

        try:
            quant_model = apply_quantization(copy.deepcopy(fp32_model), config, calib_loader, qengine)
        except Exception as err:
            print(f"Quantization failed for config '{config['name']}': {err}")
            continue
        quant_model.eval()

        quant_name = config["name"]
        quant_path = os.path.join(args.save_dir, f"nafnet_small_{args.dataset}_{quant_name}.pth")
        size_mb = save_checkpoint(quant_model, quant_path, config)
        quant_counts = count_quantized_modules(quant_model)

        metrics = evaluate_model(
            model=quant_model,
            dataset_type=args.dataset,
            sample_count=args.samples,
            seed=args.seed,
            device=device,
        )

        elapsed = time.time() - start
        results.append(
            {
                "config": config,
                "path": quant_path,
                "size_mb": size_mb,
                "metrics": metrics,
                "time_sec": elapsed,
                "quant_counts": quant_counts,
            }
        )

        print(f"Saved quantized model to {quant_path} ({size_mb:.2f} MB)")
        print(f"Quantized modules: {quant_counts}")
        print(
            f"Eval complete in {elapsed:.2f}s | PSNR {metrics['avg_psnr']:.2f} dB | "
            f"SSIM {metrics['avg_ssim']:.4f} | CPU inf {metrics['avg_inference_time_ms']:.2f} ms/img"
        )

    if results:
        print_results_table(results)
    else:
        print("No quantization runs completed.")


if __name__ == "__main__":
    main()
