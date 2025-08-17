from sb3_contrib import RecurrentPPO

from envs import frame_generator, constant_frame_generator, random_frame_generator
from envs.frame_generator_delayed import FrameGeneratorDelayed

# train constant frame
TRAIN_VIDEO_PATH = "videos/V1-0001_City Scene Layout 1 setting0001.mp4"
VIDEO_FPS = 24
FRAME_SIZE = (124,124)
N_EPISODES = 100000
target_duration = 4

env = FrameGeneratorDelayed(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH,VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES, target_duration=target_duration)
# env_constant = constant_frame_generator.ConstantFrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH,VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES)
# env_random = random_frame_generator.RandomFrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH,VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES)

# _, obs = env_random.reset()
_, obs = env.reset()
model = RecurrentPPO("CnnLstmPolicy", env, verbose=1, n_steps=1000)
# model = RecurrentPPO("CnnLstmPolicy", env_random, verbose=1, n_steps=1000)
model.learn(N_EPISODES)

model.save("models/sb3_ppo_1goal_rnn_multistop_multiseq_delayedtarget4_v2")