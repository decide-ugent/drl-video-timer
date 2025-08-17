
# mean changes in video
import cv2
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from envs.frame_generator import FrameGenerator
from envs.random_frame_generator import RandomFrameGenerator
import numpy as np

def compute_average_frame_difference(frames):
    if len(frames) < 2:
        return 0  # Not enough frames to compare

    diffs = []
    total_diff = 0
    frame_pairs = 0

    for i in range(1, len(frames)):
        # Optionally convert to grayscale
        prev_frame = frames[i-1]
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        curr_frame = frames[i]
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute pixel-wise difference
        # diff = cv2.absdiff(prev_frame_gray, curr_frame_gray)
        abs_diff = np.abs(curr_frame_gray.astype(np.float32) - prev_frame_gray.astype(np.float32))
        # mean_diff = np.mean(diff)



        # f_k(i,j) - f_{k-1}(i,j)


        # M and N: width and height of the frame
        M, N = curr_frame_gray.shape
        mafd_k = np.sum(abs_diff) / (M * N)

        total_diff += mafd_k
        frame_pairs += 1

    return total_diff / frame_pairs


if __name__ == "__main__":
    TRAIN_VIDEO_PATH = "../videos/V1-0001_City Scene Layout 1 setting0001.mp4"
    VIDEO_FPS = 24
    FRAME_SIZE = (124, 124)
    N_EPISODES = 10
    env = FrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH,VIDEO_FPS, FRAME_SIZE, n_episodes=N_EPISODES)
    _, obs = env.reset()
    print(f"MAFD for {TRAIN_VIDEO_PATH}:",compute_average_frame_difference(env.video_frames))

    # get average reward
    model = RecurrentPPO.load('models/sb3_ppo_1goal_rnn_multistop_multiseq', env=env)
    reward_list, episode_list = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True,
                                                return_episode_rewards=True)
    print("Average reward for the video", np.mean(reward_list))