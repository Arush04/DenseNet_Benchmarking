<p align="center"><h1 align="center">DENSENET_BENCHMARKING</h1></p>
<p align="center">
	<img src="https://img.shields.io/github/license/Arush04/DenseNet_Benchmarking?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Arush04/DenseNet_Benchmarking?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Arush04/DenseNet_Benchmarking?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Arush04/DenseNet_Benchmarking?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
- [ Optimization Approaches](#-optimzation-approaches)
- [ Profiling Approach Across All Variants](#-profiling-approach-across-all-variants)
- [ Results Summary](#-results-summary)

---

##  Overview

<code>❯ Benchmarking and optimizing the DenseNet architecture</code>

---


##  Project Structure

```sh
└── DenseNet_Benchmarking/
    ├── Dockerfile
    ├── benchmark_entry.py
    ├── build_and_run.sh
    └── requirements.txt
```

##  Getting Started

###  Prerequisites

Before getting started with DenseNet_Benchmarking, ensure your runtime environment meets the following requirements:

- **Programming Language:** Shell
- **Package Manager:** Pip
- **Container Runtime:** Docker


###  Installation

Install DenseNet_Benchmarking using one of the following methods:

**Build from source:**

1. Clone the DenseNet_Benchmarking repository:
```sh
❯ git clone https://github.com/Arush04/DenseNet_Benchmarking
```

2. Navigate to the project directory:
```sh
❯ cd DenseNet_Benchmarking
```

3. Enable Execution:
```sh
❯ chmod +x build_and_run.sh
```


###  Usage
Run DenseNet_Benchmarking using the following command:

```sh
❯ ./build_and_run.sh --batch-size 16 --device cuda

Other options are:  
--output-dir
--device
--batch-size
--num-workers
--iters
--warmup
--eval-batches  
```
---
## Optimization Approaches 
### Baseline PyTorch Model
The standard DenseNet121 model is loaded without any performance optimizations.  

Execution Details:
- Runs inference in eval mode to avoid gradient computation.
- No mixed-precision (AMP) or compilation applied.
- Wall-clock latency and throughput are measured using Python timers.
- PyTorch torch.profiler is used to profile kernel-level CPU and CUDA times, and memory usage (allocated/reserved VRAM).

### AMP (Automatic Mixed Precision)
AMP uses mixed precision (FP16 for compute-intensive layers, FP32 where necessary) to reduce memory usage and improve throughput on GPUs.  

Execution Details:
- Enabled via torch.cuda.amp.autocast().
- Only used if GPU (cuda) is available.
- Model remains in eval mode.
- PyTorch torch.profiler collects kernel-level metrics for AMP-enabled runs.

### TorchScript Compilation
TorchScript converts the PyTorch model into an intermediate representation (IR) that can be run independently of Python, enabling optimization and faster deployment.  

Execution Details:
- torch.jit.trace() is used to trace the model with a sample input.
- Traced model is saved and reloaded to GPU/CPU as required.
- Inference is run using the same profiling loop as baseline.
- Profiler still collects kernel-level CPU/CUDA metrics.

### ONNX + ONNX Runtime (ORT)
Export PyTorch models to ONNX format and run inference using ONNX Runtime (ORT). ORT is an optimized runtime for multiple hardware backends.  

Execution Details:
- Model exported via torch.onnx.export().
- ONNX Runtime session created with CPU or CUDA execution providers.
- Wall-clock latency and throughput measured using Python timers.
- PyTorch profiler is not used; ORT does not expose kernel-level profiling through PyTorch.
- Accuracy evaluation is optional for speed.

## Profiling Approach Across All Variants

Wall-clock Latency:
Measures real-world time per batch, including CPU-GPU transfer overhead.

Throughput:
Samples processed per second, computed as:

`throughput = batch_size / avg_latency_sec`

PyTorch Profiler Metrics:

- CPU/CUDA kernel times for low-level performance insight.
- Memory usage: peak allocated and reserved GPU memory.
- Utilization percentages: approximate kernel time as a percentage of wall-clock time.
- TensorBoard Integration:
- Profiling data is saved to TensorBoard for visual inspection.
- Kernel tables and traces are saved to disk for detailed analysis.
- CSV Output: All metrics (latency, throughput, memory, utilization, accuracy, model size) are saved in a benchmark_results.csv file.

---
## Results Summary
--batch-size 16 --device cuda --iters 200  
```sh
❯ === Benchmark Summary ===
baseline        | batch 16 | dev cuda   | lat 578.58 ms | thr 27.65 sps | top1 12.5
amp_autocast    | batch 16 | dev cuda   | lat 418.49 ms | thr 38.23 sps | top1 12.5
torchscript     | batch 16 | dev cuda   | lat 634.74 ms | thr 25.21 sps | top1 12.5
onnx_ort        | batch 16 | dev cuda   | lat 978.53 ms | thr 16.35 sps | top1 None
```
