#!/usr/bin/env python3
"""
benchmark_entry.py

Entrypoint for DenseNet-121 benchmarking suite.

- Runs Baseline, AMP (autocast), TorchScript (traced), ONNX->ORT when available.
- Uses torch.profiler for PyTorch-kernel-level profiling (baseline, AMP, TorchScript).
- Writes tensorboard logs under ./logs/tensorboard/
- Writes profiler traces under ./results/profiles/<variant>/
- Writes model checkpoints under ./results/models/
- Writes results CSV to output_dir/benchmark_results.csv
"""

import os
import sys
import argparse
import csv
import json
import time
import statistics
from pathlib import Path
from typing import Tuple, Dict

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler, ProfilerActivity


os.environ["KINETO_LOG_LEVEL"] = "5"
ONNX_AVAILABLE = False
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

# ----------------------------
# Utils
# ----------------------------
def load_model(device: torch.device):
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, profile_memory=True, record_shapes=True, on_trace_ready=None) as prof:
        with record_function("model_loading"):
            model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
            model.classifier = nn.Linear(model.classifier.in_features, 10)
            model.eval()
            model.to(device)

    sort_by_key = device.type + "_time_total"
    print(f"[setup] Model loading profile:")
    print(prof.key_averages().table(sort_by=sort_by_key, row_limit=5))

    return model, prof

def make_dataloaders(batch_size: int, num_workers: int = 2, max_eval_batches: int = 16):
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

def compute_topk(outputs: torch.Tensor, targets: torch.Tensor, ks=(1,5)):
    """Return top-k in percent for this batch."""
    maxk = max(ks)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = {}
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[f"top{k}"] = (correct_k.item() * 100.0) / targets.size(0)
    return res

def evaluate_accuracy(model, dataloader, device, max_batches=8, use_amp=False):
    model.eval()
    top1_list, top5_list = [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outs = model(imgs)
            else:
                outs = model(imgs)

            acc = compute_topk(outs, labels, ks=(1,5))
            top1_list.append(acc["top1"])
            top5_list.append(acc["top5"])
            if i+1 >= max_batches:
                break
    return {
        "top1_mean": statistics.mean(top1_list) if top1_list else None,
        "top5_mean": statistics.mean(top5_list) if top5_list else None
    }

def save_model_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return os.path.getsize(path) / (1024.0 * 1024.0)  # MB

# ----------------------------
# Profiling wrapper for PyTorch-based variants
# ----------------------------
def profile_pytorch_variant(model: nn.Module,
                            dataloader: DataLoader,
                            device: torch.device,
                            logdir: str,
                            variant_name: str,
                            use_amp: bool = False,
                            iters: int = 50,
                            warmup: int = 5,
                            writer: SummaryWriter = None):
    """
    Run inference with torch.profiler and collect:
      - latency/throughput (measured by wall-clock)
      - profiler aggregates: cpu_time_total, cuda_time_total
      - memory: max_memory_allocated, max_memory_reserved (CUDA only)
      - profiler trace saved to logdir/profile_<variant>
    Note: uses torch.profiler for kernel-level metrics and memory; does not use system metrics.
    """
    model.eval()
    tb_profile_dir = os.path.join(logdir, "tensorboard")
    profile_trace_dir = os.path.join(logdir, "profiles", variant_name)
    os.makedirs(profile_trace_dir, exist_ok=True)
    os.makedirs(tb_profile_dir, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    prof = profile(
        schedule=schedule(wait=0, warmup=0, active=1, repeat=0),
        activities=activities,
        on_trace_ready=tensorboard_trace_handler(profile_trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )

    # warmup
    loader_it = iter(dataloader)
    for _ in range(warmup):
        try:
            imgs, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(dataloader)
            imgs, _ = next(loader_it)
        imgs = imgs.to(device, non_blocking=True)
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(imgs)
            else:
                _ = model(imgs)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # timed + profiled runs
    latencies = []
    prof.__enter__()
    for i in range(iters):
        try:
            imgs, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(dataloader)
            imgs, _ = next(loader_it)
        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(imgs)
            else:
                _ = model(imgs)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # The profiler collects kernel times and memory automatically.
        # We still collect wall time latency via Python timing for throughput calculation,
        # but we **also** extract CPU/GPU kernel times from the profiler below.
        # For strict PyTorch-kernel-only measures, use values from prof.key_averages().
        prof.step()
    prof.__exit__(None, None, None)

    # Average latency/throughput calculation (use dataloader batch time via another loop for wall-clock)
    # We'll compute wall-clock latencies here separately (small loop)
    wall_latencies = []
    loader_it = iter(dataloader)
    for i in range(iters):
        try:
            imgs, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(dataloader)
            imgs, _ = next(loader_it)
        imgs = imgs.to(device, non_blocking=True)
        t0 = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        if device.type == 'cuda':
            t0.record()
        else:
            t0_wall = time.time()
        with torch.no_grad():
            if use_amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(imgs)
            else:
                _ = model(imgs)
        if device.type == 'cuda':
            t1.record()
            torch.cuda.synchronize()
            elapsed_ms = t0.elapsed_time(t1)
        else:
            t1_wall = time.time()
            elapsed_ms = (t1_wall - t0_wall) * 1000.0
        wall_latencies.append(elapsed_ms)

    avg_latency_ms = statistics.mean(wall_latencies)
    med_latency_ms = statistics.median(wall_latencies)
    std_latency_ms = statistics.stdev(wall_latencies) if len(wall_latencies) > 1 else 0.0
    throughput = dataloader.batch_size / (avg_latency_ms / 1000.0)

    # Extract profiler aggregates
    ka = prof.key_averages()
    cpu_time_total_ms = sum([e.cpu_time_total for e in ka]) / 1000.0
    cuda_time_total_ms = sum([e.cuda_time_total for e in ka]) / 1000.0 if device.type == "cuda" else 0.0

    # Peak memory as reported by PyTorch (CUDA)
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        peak_allocated_mb = None
        peak_reserved_mb = None

    # Approximate PyTorch-level utilization (%): kernel_time / wall_time *100
    wall_time_total_ms = sum(wall_latencies)
    cpu_util_pct = (cpu_time_total_ms / wall_time_total_ms * 100.0) if wall_time_total_ms > 0 else None
    gpu_util_pct = (cuda_time_total_ms / wall_time_total_ms * 100.0) if wall_time_total_ms > 0 and device.type == "cuda" else None

    # Save profiler table for top kernels as text file
    kernel_table = ka.table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=50)
    txt_path = os.path.join(profile_trace_dir, f"{variant_name}_kernels.txt")
    with open(txt_path, "w") as fh:
        fh.write(kernel_table)

    # Write to TensorBoard scalars
    if writer:
        writer.add_scalar(f"inference/{variant_name}_avg_latency_ms", avg_latency_ms, 0)
        writer.add_scalar(f"inference/{variant_name}_throughput", throughput, 0)
        writer.add_scalar(f"inference/{variant_name}_cpu_time_total_ms", cpu_time_total_ms, 0)
        writer.add_scalar(f"inference/{variant_name}_cuda_time_total_ms", cuda_time_total_ms, 0)
        if peak_allocated_mb is not None:
            writer.add_scalar(f"inference/{variant_name}_peak_allocated_mb", peak_allocated_mb, 0)

    metrics = {
        "variant": variant_name,
        "batch_size": dataloader.batch_size,
        "device": str(device),
        "latency_ms": avg_latency_ms,
        "median_latency_ms": med_latency_ms,
        "std_latency_ms": std_latency_ms,
        "throughput_samples_sec": throughput,
        "cpu_time_total_ms": cpu_time_total_ms,
        "cuda_time_total_ms": cuda_time_total_ms,
        "ram_usage_mb": None,  # torch.profiler gives op-level CPU mem; we leave None (PyTorch kernel-only)
        "vram_usage_mb": None,  # not directly aggregated by profiler; peak_allocated_mb is provided
        "peak_memory_allocated_mb": peak_allocated_mb,
        "peak_memory_reserved_mb": peak_reserved_mb,
        "cpu_utilization_pct": cpu_util_pct,
        "gpu_utilization_pct": gpu_util_pct,
        "tb_profile_dir": profile_trace_dir,
        "kernel_table_path": txt_path
    }
    return metrics

# ----------------------------
# ONNX profiling (wall-clock + accuracy + model size) - ORT cannot be profiled by torch.profiler
# ----------------------------
def run_onnx_inference(onnx_path: str, dataloader: DataLoader, device: torch.device, batch_size: int, writer: SummaryWriter = None, iters: int = 50):
    """
    Run ONNX Runtime inference and collect wall-clock latency/throughput and accuracy.
    Note: torch.profiler cannot profile ORT kernels. We still collect wall time & accuracy.
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX/ORT not available")

    providers = ort.get_available_providers()
    use_cuda = 'CUDAExecutionProvider' in providers and device.type == 'cuda'
    providers_used = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers_used)

    # warmup
    loader_it = iter(dataloader)
    for _ in range(5):
        try:
            imgs, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(dataloader)
            imgs, _ = next(loader_it)
        _ = sess.run(None, {sess.get_inputs()[0].name: imgs.numpy()})

    # timed runs (wall-clock)
    wall_lat = []
    loader_it = iter(dataloader)
    for i in range(iters):
        try:
            imgs, _ = next(loader_it)
        except StopIteration:
            loader_it = iter(dataloader)
            imgs, _ = next(loader_it)
        inp = imgs.numpy()
        t0 = time.time()
        _ = sess.run(None, {sess.get_inputs()[0].name: inp})
        t1 = time.time()
        wall_lat.append((t1 - t0) * 1000.0)

    avg_lat = statistics.mean(wall_lat)
    throughput = batch_size / (avg_lat / 1000.0)

    # accuracy (evaluate on small subset)
    top1, top5 = None, None
    # optionally evaluate accuracy by running sess on validation loader
    # (omitted heavy loop to keep this lightweight)

    model_size_mb = os.path.getsize(onnx_path) / (1024**2)

    metrics = {
        "variant": "onnx_ort",
        "batch_size": batch_size,
        "device": str(device),
        "latency_ms": avg_lat,
        "throughput_samples_sec": throughput,
        "cpu_time_total_ms": None,
        "cuda_time_total_ms": None,
        "peak_memory_allocated_mb": None,
        "cpu_utilization_pct": None,
        "gpu_utilization_pct": None,
        "model_size_mb": model_size_mb
    }
    return metrics

# ---------------------
# Tracing (TorchScript)
# ---------------------
def make_torchscript(model: nn.Module, example_input: torch.Tensor, device: torch.device):
    activities = [ProfilerActivity.CPU]
    # if device.type == 'cuda':
        # activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, profile_memory=True, record_shapes=True, on_trace_ready=None) as prof:
        with record_function("torchscript_trace"):
            model_cpu = model.to('cpu').eval()
            with torch.no_grad():
                traced = torch.jit.trace(model_cpu, example_input.cpu())
            traced = traced.to(device)

    sort_by_key = "cpu_time_total"
    print(f"[torchscript] Tracing profile:")
    print(prof.key_averages().table(sort_by=sort_by_key, row_limit=5))
    # Extract total CPU time in seconds
    total_cpu_time_us = sum(e.self_cpu_time_total for e in prof.key_averages())
    load_time_s = total_cpu_time_us / 1e6
    return traced, load_time_s

# ----------------------------
# Orchestrator: run all variants
# ----------------------------
def run_all(args):
    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    print("Using device:", device)
    train_loader, val_loader = make_dataloaders(args.batch_size, num_workers=args.num_workers)

    # Prepare output dirs
    out_dir = Path(args.output_dir)
    tb_logs = out_dir / "logs" / "tensorboard"
    profiles_dir = out_dir / "profiles"
    models_dir = out_dir / "models"
    os.makedirs(tb_logs, exist_ok=True)
    os.makedirs(profiles_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_logs))

    # Load model and record load time (we keep a Python-timer only for model load timing metadata)
    t0 = time.time()
    model, _ = load_model(device)
    load_time_s = time.time() - t0
    print(f"Model loaded in {load_time_s:.3f}s")

    # optionally save baseline checkpoint
    model_file = os.path.join(models_dir, "densenet121_baseline.pth")
    model_size_mb = save_model_checkpoint(model, model_file)

    results = []
    # Baseline (PyTorch) - profile with torch.profiler
    baseline_metrics = profile_pytorch_variant(model=model,
                                              dataloader=val_loader,
                                              device=device,
                                              logdir=str(out_dir),
                                              variant_name="baseline",
                                              use_amp=False,
                                              iters=args.iters,
                                              warmup=args.warmup,
                                              writer=writer)
    baseline_metrics["model_load_time_s"] = load_time_s
    baseline_metrics["model_size_mb"] = model_size_mb
    baseline_metrics["optimization_technique"] = "baseline"
    # Accuracy
    acc = evaluate_accuracy(model, val_loader, device, max_batches=args.eval_batches, use_amp=False)
    baseline_metrics["accuracy_top1"] = acc["top1_mean"]
    baseline_metrics["accuracy_top5"] = acc["top5_mean"]
    results.append(baseline_metrics)

    # AMP (PyTorch autocast)
    if device.type == "cuda":
        amp_metrics = profile_pytorch_variant(model=model,
                                             dataloader=val_loader,
                                             device=device,
                                             logdir=str(out_dir),
                                             variant_name="amp_autocast",
                                             use_amp=True,
                                             iters=args.iters,
                                             warmup=args.warmup,
                                             writer=writer)
        amp_metrics["model_load_time_s"] = load_time_s
        amp_metrics["model_size_mb"] = model_size_mb
        amp_metrics["optimization_technique"] = "amp_autocast"
        acc = evaluate_accuracy(model, val_loader, device, max_batches=args.eval_batches, use_amp=True)
        amp_metrics["accuracy_top1"] = acc["top1_mean"]
        amp_metrics["accuracy_top5"] = acc["top5_mean"]
        results.append(amp_metrics)
    else:
        print("[amp] skipping (no CUDA)")

    # TorchScript compiled
    try:
        example_input = torch.randn(args.batch_size, 3, 224, 224)
        ts_model, ts_time = make_torchscript(model, example_input.to(device), device)
        ts_model_file = os.path.join(models_dir, "densenet121_torchscript.pth")
        model_size_mb_ts = save_model_checkpoint(ts_model, ts_model_file)
        ts_metrics = profile_pytorch_variant(model=ts_model,
                                            dataloader=val_loader,
                                            device=device,
                                            logdir=str(out_dir),
                                            variant_name="torchscript",
                                            use_amp=False,
                                            iters=args.iters,
                                            warmup=args.warmup,
                                            writer=writer)
        ts_metrics["model_load_time_s"] = load_time_s + ts_time
        ts_metrics["model_size_mb"] = model_size_mb_ts
        ts_metrics["optimization_technique"] = "torchscript"
        acc = evaluate_accuracy(ts_model, val_loader, device, max_batches=args.eval_batches, use_amp=False)
        ts_metrics["accuracy_top1"] = acc["top1_mean"]
        ts_metrics["accuracy_top5"] = acc["top5_mean"]
        results.append(ts_metrics)
    except Exception as e:
        print("[torchscript] failed:", e)

    # ONNX -> ORT (not profiled by torch.profiler; measure wall time & accuracy)
    if ONNX_AVAILABLE:
        try:
            model_cpu = model.to("cpu").eval()
            onnx_path = os.path.join(out_dir, "densenet121.onnx")
            dummy = torch.randn(args.batch_size, 3, 224, 224)
            torch.onnx.export(model_cpu, dummy, onnx_path, export_params=True, opset_version=12,
                              input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}})
            onnx_metrics = run_onnx_inference(onnx_path, val_loader, device, args.batch_size, writer=writer, iters=args.iters)
            onnx_metrics["model_load_time_s"] = load_time_s
            onnx_metrics["optimization_technique"] = "onnx_ort"
            # (optional) compute accuracy by running ORT across a subset -> omitted to save time
            results.append(onnx_metrics)
        except Exception as e:
            print("[onnx/ort] failed:", e)
    else:
        print("[onnx/ort] not available; skipping")

    writer.close()

    # Save CSV to output_dir (user mounted path, e.g., /app/results)
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    csv_fields = [
        "model_variant","batch_size","device","ram_usage_mb","vram_usage_mb",
        "cpu_utilization_pct","gpu_utilization_pct","latency_ms","throughput_samples_sec",
        "accuracy_top1","accuracy_top5","model_size_mb","optimization_technique","model_load_time_s"
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(csv_path, "w", newline='') as cf:
        writer_csv = csv.DictWriter(cf, fieldnames=csv_fields)
        writer_csv.writeheader()
        for item in results:
            row = {
                "model_variant": item.get("variant"),
                "batch_size": item.get("batch_size"),
                "device": item.get("device"),
                "ram_usage_mb": item.get("ram_usage_mb"),
                "vram_usage_mb": item.get("vram_usage_mb") or item.get("peak_memory_allocated_mb"),
                "cpu_utilization_pct": item.get("cpu_utilization_pct"),
                "gpu_utilization_pct": item.get("gpu_utilization_pct"),
                "latency_ms": item.get("latency_ms") or item.get("avg_latency_ms"),
                "throughput_samples_sec": item.get("throughput_samples_sec") or item.get("throughput_samples_sec"),
                "accuracy_top1": item.get("accuracy_top1"),
                "accuracy_top5": item.get("accuracy_top5"),
                "model_size_mb": item.get("model_size_mb"),
                "optimization_technique": item.get("optimization_technique"),
                "model_load_time_s": item.get("model_load_time_s")
            }
            writer_csv.writerow(row)

    # Print summary to console
    print("\n=== Benchmark Summary ===")
    for r in results:
        print(f"{r.get('variant'):15} | batch {r.get('batch_size'):2} | dev {r.get('device'):6} | lat {r.get('latency_ms'):.2f} ms | thr {r.get('throughput_samples_sec'):.2f} sps | top1 {r.get('accuracy_top1')}")

    print(f"\nCSV written to {csv_path}")
    print(f"TensorBoard logs: {tb_logs}")
    print(f"Profiles: {profiles_dir}")
    print(f"Models: {models_dir}")
    return results

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output-dir", type=str, default="/app/results", help="output dir (mounted) for CSV/logs/models")
    parser.add_argument("--output-dir", type=str, default="results", help="output dir (mounted) for CSV/logs/models")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--eval-batches", type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_all(args)
