# Isaac Sim side of the comparison

This directory builds the same wire, fork and board in **Isaac Sim (PhysX)** and runs the
same benchmark rigs as MuJoCo, so `harness_bench` can score both with one analysis.

It is a separate directory on purpose: Isaac Sim needs its own Python 3.11 environment and
does not belong in the colcon workspace (there is no `package.xml`, so colcon ignores it).

---

## 1. What runs where

| | MuJoCo | Isaac Sim |
|---|---|---|
| wire model | Cosserat rod (cable plugin) | capsule chain, D6 joint drives with `k = EI / L_segment` |
| solver | implicit, elliptic friction cone | PhysX TGS, 32 position iterations |
| runs on | any CPU, no GPU needed | RTX GPU (see below) |
| in this repo | `harness_core/rigs_mujoco.py` | `isaac/harness_isaac/rigs_isaac.py` |

The two wire models are *not* the same physics. That is the point of measuring them
against the same analytic references (elastica, catenary) instead of against each other.

## 2. Your machine

The laptop this was written for is an ASUS TUF F15 (i9-13900H, 32 GB RAM, RTX 4060 Laptop
with 8 GB VRAM, Windows 11). Isaac Sim 5.1/6.0 asks for an RTX card with **16 GB** VRAM as
the minimum, so 8 GB is below spec. In practice the scenes here are tiny (a few dozen
rigid bodies, no photoreal rendering), so they should run; what you will not be able to do
on this machine is large-scale parallel training or heavy RTX rendering.

Run everything **headless** (the rigs already do) and keep an eye on RAM: Isaac Sim's first
start-up compiles shaders and pulls assets, which takes a long time and a few GB of disk.

## 3. Install (Windows 11, native)

```powershell
py -3.11 -m venv %USERPROFILE%\isaacsim-venv
%USERPROFILE%\isaacsim-venv\Scripts\activate
python -m pip install --upgrade pip
set OMNI_KIT_ACCEPT_EULA=YES
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install numpy pyyaml
```

Then, from this directory:

```powershell
python -m harness_isaac.smoke_test          # first run: builds one wire, lets it droop
python run_isaac_benchmarks.py --out results\isaac.json --quick
python run_isaac_benchmarks.py --out results\isaac.json
```

The smoke test is deliberately small: if the import path, the GPU or the USD API is wrong,
it fails in ten lines instead of halfway through the suite.

WSL2 also works for many people (CUDA comes through the Windows driver), but NVIDIA does
not list it as supported, and the ROS workspace does not need to be in the same shell, so
the native Windows route is the one to try first.

## 4. Comparing

`isaac.json` and the MuJoCo run are combined by the same tool:

```bash
python3 -m harness_bench.compare results/mujoco.json results/isaac.json --out report
```

which writes `report/report.md` plus the figures.

## 5. When something breaks

Isaac Sim's Python API moved packages between 4.x (`omni.isaac.core`) and 5.x
(`isaacsim.core.api`), and a few of the USD physics details here are written from the
specification rather than from a run on this machine (there is no GPU in the environment
this was developed in). `compat.py` already tries both import paths. If a call fails, the
useful thing to send back is the full traceback plus the output of

```powershell
python -c "import isaacsim, sys; print(isaacsim.__version__, sys.version)"
```

Most likely candidates, in order: the drive stiffness unit convention (USD angular drives
are per **degree**, which `scene_usd.py` converts for), whether reading poses needs the
Fabric/USD flag set, and the exact name of the articulation force reader.
