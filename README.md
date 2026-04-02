# ANYmal RL Policy Control

Interactive deployment of a reinforcement-learning locomotion policy for the **ANYmal C** quadruped robot, simulated in the [Newton](https://github.com/nvidia/newton) physics engine (MuJoCo solver backend).

The policy was trained in Isaac Lab and is executed here at **50 Hz** (policy) / **200 Hz** (physics) with keyboard-driven velocity commands.

## Features

- **Two control modes** selectable in the source code:
  - **Option A — POSITION mode:** Newton's built-in PD controller applies torques at the physics rate.
  - **Option B — EFFORT mode (default):** A custom `pd_control()` function computes and sends joint torques directly, running inside the decimation loop at the full 200 Hz physics rate.
- **CPU and CUDA support** — automatically detects the device; can be forced via CLI.
- **PhysX-trained policy** support via `--physx` flag with automatic joint-name remapping.
- **Inference timing** — per-step wall-clock (CPU) or CUDA-event timing of the full control pipeline (observation + policy + PD × 4 decimation steps), saved as `.npy` on exit.
- **Torque logging** — optional per-step torque recording (commented out by default to avoid GPU sync overhead).
- **Real-time rate limiting** — the loop sleeps to match wall-clock time.
- **CUDA graph acceleration** — when running on CUDA with memory pools enabled, physics stepping is captured and replayed as a CUDA graph.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.1+ drivers (for GPU mode; CPU-only works without)

### Setup

Create and activate a virtual environment, then install the dependencies:

```bash
# Create virtual environment
python -m venv env_newton

# Activate (Windows PowerShell)
.\env_newton\Scripts\Activate.ps1

# Activate (Linux / macOS)
source env_newton/bin/activate
```

Install PyTorch with CUDA 12.1 support (adjust the index URL for your CUDA version or use `cpu` for CPU-only):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the remaining dependencies:

```bash
pip install newton-physics pyyaml numpy
```

`newton-physics` pulls in its required dependencies automatically: `warp-lang`, `mujoco`, `mujoco-warp`, `usd-core`, and others.

> Robot assets (USD model and RL policy weights) are downloaded automatically on first run via `newton.utils.download_asset()`.

## Usage

```bash
# Run with GPU (default)
python anymal_robot_policy.py

# Run on CPU
python anymal_robot_policy.py --device cpu

# Run the PhysX-trained policy instead of the MuJoCo-Warp one
python anymal_robot_policy.py --physx

# All Newton example flags are supported (--test, --num_frames, etc.)
python anymal_robot_policy.py --help
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `I` / `K` | Forward / Backward |
| `J` / `L` | Strafe left / right |
| `U` / `O` | Rotate left / right |
| `P` | Reset robot to initial pose |
| `Space` | Pause / resume simulation |
| `Esc` | Quit |

## Project Structure

```
anymal_rl/
├── anymal_robot_policy.py       # Main script
├── anymal_robot_policy_bkp.py   # Reference/backup (Newton built-in PD, POSITION mode)
├── doc/
│   └── newton_examples_framework.md   # Newton examples framework reference
├── log_data/                    # Saved inference timing & torque logs (.npy)
└── README.md
```

## Control Modes

The script contains two selectable control paths, toggled by commenting/uncommenting blocks marked `Option A` and `Option B` in three locations: the builder config, `capture()`, and `step()`.

### Option A — POSITION mode (Newton built-in PD)

Newton receives **target joint positions** and internally applies a PD controller at every physics substep using the stiffness (`ke`) and damping (`kd`) gains from the YAML config.

### Option B — EFFORT mode (manual PD) *(active by default)*

The script sets `ke = kd = 0` and `JointTargetMode.EFFORT`, then computes torques explicitly:

$$\tau = K_p \cdot (q_{\text{target}} - q) - K_d \cdot \dot{q}$$

Torques are recomputed **inside** the decimation loop (4× per policy step) at the 200 Hz physics rate. Computing them only once at 50 Hz causes instability because the effective controller bandwidth is too low for the stiff gains.

## Inference Timing

On exit, the script prints timing statistics and saves a `.npy` array to `log_data/`:

```
[INFO] Control pipeline: 857 calls, mean = 6.3077 ms, std = 29.2025 ms
[INFO] Inference times saved to: log_data/anymal_robot_policy_0204_141139.npy
```

Each entry measures the **full control pipeline** per policy step: observation computation + policy forward pass + action rearrangement + PD torque computation × 4 decimation steps. The physics simulation itself is **not** included.

**CPU vs GPU timing notes:**
- For single-robot batch-1 inference, CPU is typically faster (~2–3 ms vs ~4–7 ms on GPU) because kernel launch overhead dominates over the negligible compute of a small MLP.
- The first step is significantly slower on both devices (CUDA context / JIT warm-up).
- GPU timing uses `torch.cuda.Event` pairs; CPU timing uses `time.perf_counter()`.

Load and analyze saved logs:

```python
import numpy as np

t = np.load("log_data/anymal_robot_policy_0204_141139.npy")
print(f"mean = {t.mean():.2f} ms, median = {np.median(t):.2f} ms, std = {t.std():.2f} ms")
```

## License

See individual dependency licenses (Newton, PyTorch, Warp).
