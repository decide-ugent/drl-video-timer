# =============================================================================
# DRL Training Script — trains a recurrent PPO agent on the interval timing tasks
# =============================================================================

from sb3_contrib import RecurrentPPO

from rl_environments.frame_generator import FrameGenerator
from rl_environments.constant_frame_generator import ConstantFrameGenerator
from rl_environments.random_frame_generator import RandomFrameGenerator

# -------------------------------------------------------------------------
# Environment and training configuration
# ------------------------------------------------------------------------
TRAIN_VIDEO_PATH = "videos/V1-0001_City Scene Layout 1 setting0001.mp4"
VIDEO_FPS = 24
FRAME_SIZE = (124, 124)
N_EPISODES = 50000
target_duration = 3

# -------------------------------------------------------------------------
# Initialize the environment
# -------------------------------------------------------------------------

env = FrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH, VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES,
                     target_duration=target_duration)
# Uncomment below to switch environments:
# env_constant = ConstantFrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH,VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES)
# env_random = RandomFrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH,VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES)

# -------------------------------------------------------------------------
# Reset environment to get the initial observation
# -------------------------------------------------------------------------

# _, obs = env_random.reset()
_, obs = env.reset()

# -------------------------------------------------------------------------
# Initialize Recurrent PPO agent
# -------------------------------------------------------------------------

model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, n_steps=1000)
# model = RecurrentPPO("CnnLstmPolicy", env_random, verbose=1, n_steps=1000)

# -------------------------------------------------------------------------
# Train and save the model
# -------------------------------------------------------------------------

model.learn(N_EPISODES)

model.save("models/sb3_ppo_1goal_rnn_multistop_multiseq_target3")
