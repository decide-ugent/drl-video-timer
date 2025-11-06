import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import os
from typing import Tuple
import cv2


# =============================================================================
# Custom Gym environment: ConstantFrameGenerator
# This environment presents the same video frame to the agent at each timestep.
# It is used to check how the visual input affects agent's timing behaviour
# =============================================================================
class ConstantFrameGenerator(gym.Env):

    def __init__(self, train_video_path: str, test_video_folder: str, video_fps: int, frame_size: Tuple[int, int],
                 n_actions: int = 2,
                 train_env=True, n_episodes: int = 1000, tolerance: int = 1, target_duration: int = 4):
        super().__init__()

        # Environment parameters
        self.n_actions = n_actions
        self.frame_size = frame_size

        self.video_fps = video_fps
        self.episode_num = -1
        self.n_episodes = n_episodes
        self.train_video_path = train_video_path
        self.test_video_folder = test_video_folder

        self.target_duration = target_duration
        if train_env:
            self.video_path = self.train_video_path
        else:
            self.video_path = random.choice(os.listdir(self.test_video_folder))

        # Load a single frame from the video and repeat it to simulate a constant frame
        self.fullvideo_frames = self.load_single_frame(self.video_path, self.video_fps)
        if np.sum(self.fullvideo_frames[0]) < 5:
            self.fullvideo_frames = [self.fullvideo_frames[0] + 255] * 100
        else:
            self.fullvideo_frames = [self.fullvideo_frames[0]] * 100

        self.action_reward_history = {}
        self.tolerance = tolerance

        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)



    # -------------------------------------------------------------------------
    # Load a single frame from the video
    # -------------------------------------------------------------------------
    def load_single_frame(self, video_path: str, fps: int):

        cap = cv2.VideoCapture(video_path)
        video_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            video_frames.append(frame)
            break
        cap.release()

        return video_frames

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.save_action_reward = False
        self.ep_actions = []
        self.ep_rewards = []
        self.episode_num += 1
        if self.episode_num % 500 == 0:
            self.save_action_reward = True

        self.current_frame_index = 0
        start_frame = 0

        self.video_frames = self.fullvideo_frames[start_frame:start_frame + self.target_duration * 5]
        self.state = self.video_frames[self.current_frame_index]
        self.info = {"action": 0, "current_frame_index": 0, "timestep": self.timestep}
        return self.state, self.info

    def step(self, action):
        self.timestep += 1
        self.current_frame_index += 1

        # Update info dictionary
        self.info["action"] = action
        self.info["current_frame_index"] = self.current_frame_index
        self.info["timestep"] = self.timestep

        reward = self.compute_reward2(self.target_duration, self.info)
        terminated = self.compute_terminated2(self.target_duration, self.info)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

        if self.current_frame_index % self.target_duration == 0:
            self.timestep = 0

        if not terminated:
            self.state = self.video_frames[self.current_frame_index]

        else:  # end of video reached
            self.state = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

        if self.save_action_reward:
            # print("action", self.ep_actions)
            self.action_reward_history[self.episode_num // 500] = {"action": [], "reward": []}
            self.action_reward_history[self.episode_num // 500]["action"] = self.ep_actions
            # print(self.action_reward_history[self.episode_num//500]["action"])
            self.action_reward_history[self.episode_num // 500]["reward"] = self.ep_rewards

        return self.state, reward, terminated, False, self.info

    def compute_reward2(self, target_duration, info):

        # Reward function:
        # - If action=0 ("Go" to next frame) and timestep < target, reward=0.
        # - If action=0 and timestep >= target, reward=-1.
        # - If action=1 (mark "Interval") and timestep==target, reward=1, else reward=-1.

        if info["action"] == 0:
            if info["timestep"] < target_duration:
                reward = 0

            else:  # target duration reached
                reward = -1

        else:

            if info["timestep"] == target_duration:
                reward = 1
            else:
                reward = -1

        return reward

    def compute_terminated2(self, target_duration, info):

        # Episode ends when we have processed target_duration * 5 frames.

        if info["current_frame_index"] < target_duration * 5:
            terminated = False

        else:  # end of video reached
            terminated = True
        # else:
        #   terminated = True
        return terminated

    def compute_truncated2(self, achievec_goal, desired_goal, info):
        return False
