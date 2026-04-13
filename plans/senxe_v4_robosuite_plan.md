# Project Senxe v4.0 — RoboSuite PegInHole Architecture Plan

## Target: Alon Loeffler (Cortical Labs)
## Demo: "CL1 Bio vs PPO vs Random sample-efficiency benchmark on industrial assembly with real force feedback"

---

## 1. File Structure

```
├── senxe_demo.py              # v3.0 (UNCHANGED — FetchPickAndPlace)
├── senxe_demo_robosuite.py    # v4.0 (NEW — RoboSuite PegInHole)
├── README.md                  # Updated to v4.0
├── requirements.txt           # Updated with robosuite
├── test_mujoco.py
└── plans/
    └── senxe_v4_robosuite_plan.md
```

---

## 2. Key Architectural Changes (v3.0 → v4.0)

### 2.1 Environment: FetchPickAndPlace → RoboSuite PegInHole

```python
# v3.0
env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", reward_type="dense")

# v4.0
import robosuite as suite
from robosuite.wrappers import GymWrapper
env = suite.make("PegInHole", robots="Panda",
                 has_renderer=False, use_camera_obs=False,
                 has_offscreen_renderer=True, render_camera="frontview")
env = GymWrapper(env)
```

### 2.2 VIE Encoding: Add Native Force/Torque Channels

v3.0 channel layout:
```
CH  0-7     Distance/Pressure    (rate coding)
CH  8-15    Visual Brightness    (Doom-style pixel rate coding)
CH 16-31    Velocity             (temporal coding, traveling waves)
CH 32-47    Position             (absolute grip position)
CH 48-55    Goal Direction       (delta vector guidance)
CH 56-63    Object Direction     (object-to-grip delta)
```

v4.0 channel layout (force/torque replaces visual brightness + object direction):
```
CH  0-15    Force feedback       (rate coding — native force sensor)
            # Native force/torque feedback (Doom-style visual + tactile)
CH 16-31    Torque/friction      (traveling waves — native torque sensor)
CH 32-47    Position encoding    (absolute end-effector position)
CH 48-55    Goal Direction       (delta vector to insertion target)
CH 56-63    Insertion Depth      (peg depth progress encoding)
```

### 2.3 Observation Extraction

RoboSuite PegInHole observations include:
- `robot0_eef_pos` — end-effector position (3D)
- `robot0_eef_vel_lin` — linear velocity (3D)
- `robot0_force` — force sensor readings (3D or 6D)
- `robot0_torque` — torque sensor readings
- Joint positions, velocities, etc.

Key extraction:
```python
# Force/torque from native sensors
force = obs_dict.get("robot0_eef_force", obs_dict.get("robot0_force", np.zeros(3)))
torque = obs_dict.get("robot0_eef_torque", obs_dict.get("robot0_torque", np.zeros(3)))
eef_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))
eef_vel = obs_dict.get("robot0_eef_vel_lin", np.zeros(3))
```

### 2.4 Success Metric (Dual Criteria)

```python
# Peg inserted: depth > threshold
peg_depth = compute_insertion_depth(obs)
inserted = peg_depth > INSERTION_DEPTH_THRESHOLD  # e.g., 0.02m

# Force safety: not too high (avoid damage)
force_mag = np.linalg.norm(force)
force_safe = force_mag < FORCE_SAFETY_THRESHOLD  # e.g., 20N

# Combined success
success = inserted and force_safe
```

### 2.5 Action Space Adaptation

v3.0: 4D action (x, y, z, gripper)
v4.0: RoboSuite PegInHole action dim varies (typically 7D for Panda — joint velocities or OSC)

AntagonisticDecoder adapts automatically via `env.action_space.shape[0]`.

### 2.6 Learning Curve: Add Force Safety Rate

```python
# Two secondary Y-axes or combined subplot:
# 1. Success Rate (%) — peg insertion success
# 2. Force Safety Rate (%) — episodes where max force stayed within safe range
```

### 2.7 Video Outputs

```
cl1_peginhole.mp4          — CL1 bio-agent PegInHole task
side_by_side_robosuite.mp4 — CL1 vs PPO side-by-side comparison
learning_curve_robosuite.png — Learning curves + Success Rate + Force Safety Rate
```

---

## 3. Components to Keep 100% Unchanged (Biological Core)

| Component | Lines in v3.0 | Status |
|---|---|---|
| MockChannelSet / MockStimDesign / MockBurstDesign | 88-104 | Copy verbatim |
| MockNeurons (64-ch MEA simulator + metabolic health) | 106-167 | Copy verbatim |
| cl_open() / mock_open() | 169-188 | Copy verbatim |
| CL SDK auto-detect (try import cl) | 78-83 | Copy verbatim |
| warmup_calibration() | 199-240 | Copy verbatim |
| AntagonisticDecoder (flexor/extensor + EMA) | 397-455 | Copy verbatim |
| PDI (velocity + acceleration variance) | 467-491 | Copy verbatim |
| Dopamine Injection (_dopamine_inject) | 751-764 | Copy verbatim |
| Metabolic Guardrail (in MockNeurons) | 121-149 | Copy verbatim |
| Overlay functions (text, bar chart, reward curve) | 506-690 | Adapt for force display |

---

## 4. Components to Adapt

| Component | Change |
|---|---|
| VIE.encode() | Replace visual brightness with force rate coding (CH 0-15), replace object direction with torque traveling waves (CH 16-31), add insertion depth encoding |
| CL1Agent.run_episode() | Extract RoboSuite obs format, compute insertion depth + force safety, dual success metric |
| train_ppo_baseline() | Use RoboSuite env with GymWrapper + FlattenObservation |
| run_random_baseline() | Use RoboSuite env with GymWrapper |
| plot_learning_curves() | Add Force Safety Rate as additional metric |
| Overlay text | Show Force/Torque/Depth instead of Gripper status |
| Configuration constants | New env name, thresholds, video filenames |

---

## 5. Configuration Constants (v4.0)

```python
ENV_NAME          = "PegInHole"
ROBOT             = "Panda"
CL1_EPISODES      = 200          # Fast test for M4 Pro
CL1_MAX_STEPS     = 200          # PegInHole needs more steps
PPO_TIMESTEPS     = 20_000       # 20k steps
PPO_EVAL_EPS      = 200
RANDOM_EPISODES   = 200
RECORD_LAST_N     = 80
WARMUP_SECONDS    = 10
ACTION_SCALE      = 0.25         # Finer for precision assembly
INSERTION_DEPTH_THRESHOLD = 0.02 # meters
FORCE_SAFETY_THRESHOLD    = 20.0 # Newtons
VIDEO_CL1         = "cl1_peginhole.mp4"
VIDEO_SIDE        = "side_by_side_robosuite.mp4"
PLOT_FILE         = "learning_curve_robosuite.png"
```

---

## 6. Run Commands

```bash
pip install robosuite
export MUJOCO_GL=glfw
python senxe_demo_robosuite.py
```
