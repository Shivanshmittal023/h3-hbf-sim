# Running the Full H3 Experiments on a Large Machine

The laptop flow in the [README](../README.md) proves the pipeline is correct.
Reproducing the paper's results needs real GPUs (to *generate* traces), a lot of
disk, and a lot of RAM. This document covers what to provision and how to set it
up, including on HPC clusters with no Docker.

---

## 1. Why a big machine is needed at all

Accel-Sim is **trace-driven**: simulation itself needs no GPU. But the traces
have to come from somewhere, and NVBit instruments a *real* CUDA execution.

```
  [ real GPU + NVBit ]  →  traces  →  [ any CPU machine ]  →  results
     needs A100/H100        TBs         needs RAM + disk
```

So there are two distinct machine profiles, and they need not be the same host.

---

## 2. Hardware requirements

### Profile A — trace generation (needs real GPUs)

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU | 1× A100 40 GB | 8× H100 80 GB | Must hold the model shard being traced |
| CUDA | 12.8 | 12.8 | Matches the Accel-Sim toolchain |
| Compute capability | ≥ 7.5 | ≥ 9.0 | NVBit requirement; 9.0 for Hopper SASS |
| Host RAM | 128 GB | 512 GB | NVBit buffers traces in host memory |
| Disk | 5 TB | 50 TB NVMe | See the trace-size table below |
| Driver | ≥ 570 | ≥ 570 | Required by CUDA 12.8 |

Llama 3.1 405B at FP8 is ~406 GB of weights. Tracing needs the weights resident
across the GPUs — 8× H100 80 GB (640 GB) fits with room for KV cache.

### Profile B — simulation (no GPU)

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| CPU | 16 cores | 64+ cores | One core per concurrent kernel simulation |
| RAM | 64 GB | 256 GB | GPGPU-Sim holds warp state for 132 SMs |
| Disk | 5 TB | 50 TB | Traces plus per-kernel outputs |
| OS | Ubuntu 22.04/24.04 | Ubuntu 24.04 | Matches the container |

**Simulation is single-threaded per kernel but embarrassingly parallel across
kernels.** Use `util/job_launching/run_simulations.py -l sbatch` on Slurm.

---

## 3. Disk budget

Trace size scales with *instructions executed*, which scales with sequence
length. Measured from the synthetic generator and extrapolated:

| Experiment | Layers | Sequence | GPUs | Raw traces | Compressed `.tracez` |
|---|---:|---:|---:|---:|---:|
| Smoke (this repo) | 4 | 1K | 0 | 6 MiB | — |
| Llama 405B, 1 layer | 1 | 1M | 1 | ~40 GB | ~8 GB |
| Llama 405B, full | 126 | 1M | 8 | ~2–5 TB | ~0.5–1 TB |
| Llama 405B, full | 126 | 10M | 32 | ~20–50 TB | ~5–10 TB |

**Always use compressed traces.** `libzstd-dev` is already in the container;
`.tracez` is roughly 5× smaller and Accel-Sim reads it natively.

Two ways to cut the budget dramatically:

1. **Trace one layer, replay 126×.** Transformer layers are structurally
   identical, so a single traced layer plus an address offset per layer captures
   the memory behaviour at ~1/126 the disk. This is the recommended approach and
   what `tools/generate_synthetic_trace.py` already models.
2. **Sample kernels.** `run_simulations.py` accepts a kernel subset; the steady
   state of decode is highly repetitive.

---

## 4. Setup on a Linux machine with Docker

Identical to the laptop flow, minus the emulation penalty:

```bash
git clone --recurse-submodules https://github.com/Shivanshmittal023/h3-hbf-sim.git h3-hbf-sim
cd h3-sim
docker compose build
docker compose run --rm h3sim bash

./scripts/install_hbf.sh
./scripts/apply_gpgpu_sim_patch.sh
source ./gpu-simulator/setup_environment.sh release
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release
cmake --build ./gpu-simulator/build/release -j$(nproc)      # ~5 min native
cmake --install ./gpu-simulator/build/release
```

For NVBit tracing the container additionally needs GPU access:

```bash
docker run --gpus all --runtime=nvidia -v $PWD:/workspace/h3-sim -it h3sim:cuda12.8-ubuntu24.04 bash
```
(requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

---

## 5. Setup on an HPC cluster without Docker

Many clusters disallow Docker. Everything here builds natively.

```bash
# 1. Toolchain (module names vary by site)
module load cuda/12.8 gcc/13 cmake/3.27 python/3.12

# 2. Dependencies without root -- via conda, or ask for a site install of:
#    build-essential xutils-dev bison flex zlib1g-dev libzstd-dev
#    libssl-dev libxml2-dev libboost-all-dev libglu1-mesa-dev python3-dev
conda create -n h3 -c conda-forge \
    gcc_linux-64=13 gxx_linux-64=13 cmake make bison flex \
    zlib zstd boost-cpp libxml2 openssl mesalib python=3.12
conda activate h3

python3 -m pip install -r requirements.txt

# 3. Environment
export CUDA_INSTALL_PATH=$CUDA_HOME
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

# 4. Build
git clone --recurse-submodules https://github.com/Shivanshmittal023/h3-hbf-sim.git h3-hbf-sim && cd h3-sim
./scripts/install_hbf.sh
./scripts/apply_gpgpu_sim_patch.sh
source ./gpu-simulator/setup_environment.sh release
cmake -S ./gpu-simulator/ -B ./gpu-simulator/build/release
cmake --build ./gpu-simulator/build/release -j$(nproc)
cmake --install ./gpu-simulator/build/release

# 5. Verify without running anything large
./scripts/run_unit_tests.sh          # 166 assertions, seconds
```

Notes:
- **Apptainer/Singularity** can convert the image:
  `apptainer build h3sim.sif docker-daemon://h3sim:cuda12.8-ubuntu24.04`
- **CMake fetches Ramulator's dependencies at configure time** (yaml-cpp, fmt,
  nanobind). On an air-gapped node, configure once on a login node with network
  access, or pre-populate `ramulator2/ext/`.
- Put traces on **scratch/parallel filesystem**, never in `$HOME` (quotas).

---

## 6. Generating Llama traces with NVBit

Requires a machine with real GPUs (Profile A).

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
```

`run_hw_trace.py` only handles benchmarks declared in
`util/job_launching/apps/`. For a serving framework, use the `run.sh` wrapper,
which sets up the NVBit environment and `exec`s whatever you give it:

```bash
NVBIT_INSTRUMENTATION_ENABLED=1 \
TRACES_FOLDER=/scratch/$USER/h3-traces/llama405b-1M \
  ./util/tracer_nvbit/others/torch_hook/run.sh \
  python3 your_llama_inference.py --seq-len 1048576 --dtype fp8

# post-process raw traces into compressed .tracez
./util/tracer_nvbit/traces_processing/post-traces-processing \
    /scratch/$USER/h3-traces/llama405b-1M/kernelslist
```

**Getting H3's address map into the trace.** NVBit records whatever virtual
addresses the application uses, which will *not* match the H3 map. **This is
already solved** — `tools/classify_trace.py` builds an address map from the
trace itself:

```bash
python3 tools/classify_trace.py /scratch/$USER/h3-traces/llama405b-1M \
    -o configs/generated/llama_1m_map.yaml
```

It finds every buffer the trace touches, tallies reads and writes per buffer,
and applies H3's placement rule — large and never written goes to HBF,
everything else to HBM. The router loads the result via
`address_map.allocation_map` and decodes by buffer instead of by address range.

For a real inference trace this should classify the weight tensors and the
shared KV cache into HBF automatically, because the framework never writes them.
Check the reported `traffic share to HBF` against the expectation from
`configs/llama_405b_config.yaml`: if it is far below the ~99% the model implies,
the framework is probably writing a buffer you expected to be read-only, and
that is worth understanding before simulating.

Either way, validate before simulating:

```bash
python3 tools/validate_trace.py /scratch/$USER/h3-traces/llama405b-1M
```
`unmapped addrs` must be 0. Non-zero means the trace's addresses do not match
`configs/h3_router_config.yaml`, and every routing statistic would be garbage.

---

## 7. Running the experiments

```bash
# 1M tokens, 8 GPUs
./util/job_launching/run_simulations.py \
    -B llama405b-1M -C SM90_H100-H3 \
    -T /scratch/$USER/h3-traces/llama405b-1M -N h3-1m -l sbatch

./util/job_launching/monitor_func_test.py -v -N h3-1m
./util/job_launching/get_stats.py -B llama405b-1M -C SM90_H100-H3 | tee h3-1m.csv
```

H3 counters are emitted in GPGPU-Sim's `key = value` style, so `get_stats.py`
collects them with no modification.

Suggested sweeps (all single-parameter, all documented in
[hbf_model.md](hbf_model.md)):

| Sweep | Values | Question answered |
|---|---|---|
| `TR_NS` | 5, 10, 20, 40, 80 µs | How sensitive is H3 to NAND latency? |
| `lhb.buffer_size_mb` | 10, 20, 40, 80, 160 | Is Eq. (1) the right sizing rule? |
| planes/PC | 16, 32, 64, 128 | Where is the bandwidth cliff? |
| `lead_time_ns` | 20, 30, 45, 60, 90 µs | How much lead time is really needed? |
| `enable_prefetch_scheduler` | true/false | Buffer alone vs buffer + hints |

---

## 8. Before trusting throughput-per-watt numbers

The paper's headline is **2.69× throughput per watt**. This repo does not yet
reproduce it, and two gaps must be closed first:

1. **AccelWattch DRAM power is first-order under H3.** Reads and writes are
   real; row-command counts (`n_act`/`n_pre`) are derived, not measured. Enable
   `-power_simulation_enabled 1`, then extend
   `H3MemoryBackend::set_dram_power_stats()` to report real command counts —
   which requires `device_backend: ramulator`, since the analytic devices are
   not command-level models.
2. **HBF and HBM have different power models.** AccelWattch models DRAM, not
   NAND. The paper's 160 W/cube HBF vs 40 W/cube HBM3e must be applied as
   separate coefficients per region; the router already reports per-region
   traffic (`h3_router_bytes_hbm` / `_hbf`) to drive that.

Until both are done, treat performance results as valid and power results as
placeholders.
