"""
RoboSuite PegInHole quick visualization test.

Launches a RoboSuite PegInHole environment with a Franka Panda arm
and renders 500 random-action steps to verify MuJoCo + RoboSuite
installation.

Usage: python test_mujoco.py
"""
import robosuite as suite

# Create environment (has_renderer=True opens a 3D visualization window)
env = suite.make("PegInHole", robots="Panda", has_renderer=True)

# Reset environment
obs = env.reset()

print("RoboSuite test started — PegInHole window should be visible, running random actions...")

# Run 500 random action steps
for step in range(500):
    action = env.action_spec[0].shape[0] * [0.0]  # Zero action placeholder
    import numpy as np
    action = np.random.uniform(-1, 1, env.action_spec[0].shape[0])
    obs, reward, done, info = env.step(action)
    env.render()

    # Reset on episode termination
    if done:
        obs = env.reset()

# Cleanup
env.close()
print("Test complete — environment closed successfully.")
