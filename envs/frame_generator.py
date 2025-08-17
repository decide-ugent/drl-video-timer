
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import os
from typing import Tuple
import cv2

class FrameGenerator(gym.Env):

    def __init__(self, train_video_path: str,test_video_folder:str, video_fps: int, frame_size: Tuple[int,int] , n_actions:int=2,
                 train_env=True , n_episodes:int=1000, tolerance:int=1,target_duration:int=4):
        super().__init__()
        self.n_actions = n_actions
        self.frame_size = frame_size


        self.video_fps = video_fps
        self.episode_num = -1
        self.n_episodes = n_episodes
        self.train_video_path = train_video_path
        self.test_video_folder = test_video_folder
        self.color_map = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "white": (255, 255, 255),
            "violet": (127,0,255),
            "cyan": (255, 255, 0),
            "yellow": (0, 255, 255),
        }

        self.target_duration = target_duration
        if train_env:
            self.video_path = self.train_video_path
        else:
            self.video_path = random.choice(os.listdir(self.test_video_folder))
        self.fullvideo_frames = self.load_fullvideo(self.video_path, self.video_fps)

        self.action_reward_history = {}
        self.tolerance = tolerance

        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.frame_size[0], self.frame_size[1],3), dtype=np.uint8)

        # self.reset()

    def load_fullvideo(self, video_path:str, fps: int):

        cap = cv2.VideoCapture(video_path)
        video_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.frame_size)
            video_frames.append(frame)
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
        # self.current_color = random.choice(list(self.color_time_map.keys()))
        self.current_frame_index = 0
        start_frame = random.randint(0, len(self.fullvideo_frames)-5*self.target_duration-1)
        # print("start frame", start_frame, start_frame+30)
        self.video_frames = self.fullvideo_frames[start_frame:start_frame+self.target_duration*5]
        self.state = self.video_frames[self.current_frame_index]
        self.info = {"action":0, "current_frame_index":0, "timestep":self.timestep}
        return self.state, self.info

    def step(self, action):
        self.timestep += 1

        self.current_frame_index += 1
        self.info["action"] = action
        self.info["current_frame_index"] = self.current_frame_index
        self.info["timestep"] = self.timestep

        reward = self.compute_reward2(self.target_duration, self.info)
        terminated = self.compute_terminated2( self.target_duration, self.info)
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)

        if self.current_frame_index % self.target_duration == 0:
          self.timestep = 0

        if not terminated:
            self.state = self.video_frames[self.current_frame_index]

        else:    # end of video reached
            self.state = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)

        if self.save_action_reward:
            # print("action", self.ep_actions)
            self.action_reward_history[self.episode_num//500] ={"action":[],"reward":[]}
            self.action_reward_history[self.episode_num//500]["action"] = self.ep_actions
            # print(self.action_reward_history[self.episode_num//500]["action"])
            self.action_reward_history[self.episode_num//500]["reward"] =  self.ep_rewards


        return self.state, reward, terminated, False, self.info


    def compute_reward2(self,target_duration, info):

      if info["action"] == 0:
        if info["timestep"] < target_duration:
          reward = 0

        else:    # target duration reached
          reward = -1

      else:

        if info["timestep"] == target_duration:
          reward = 1
        else:
          reward = -1

      return reward


    def compute_terminated2(self, target_duration, info):

      if info["current_frame_index"] < target_duration*5:
        terminated = False

      else:    # end of video reached
        terminated = True
      # else:
      #   terminated = True
      return terminated



    def compute_truncated2(self, achievec_goal, desired_goal, info):
        """The environments will be truncated only if setting a time limit with max_steps which will automatically wrap the environment in a gym TimeLimit wrapper."""
        return False