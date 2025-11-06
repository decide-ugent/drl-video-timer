from sb3_contrib import RecurrentPPO

from rl_environments.frame_generator import FrameGenerator
import torch as t

from sb3model_wrappers.recurrentppo_wrapper import sb3Wrapper


class LayerOutputCalculator():

    def __init__(self, wrapped_model, model, env):
        self.wrapped_model = wrapped_model
        self.model = model
        self.env = env


    def get_layer_outputs_per_episode(self, timesteps=50):

        obs, _ = self.env.reset()

        self.dataset_train = []

        print("target time",self.env.target_duration)
        self.values = []
        self.lstm_states_actor_h = []
        self.lstm_states_actor_c = []
        self.lstm_state_critic_h = []
        self.lstm_state_critic_c = []
        self.mlp_policy_features = []
        self.mlp_value_features = []
        # observations = []
        self.actions = []

        states = None
        self.lstm_hidden = []
        self.lstm_cell = []
        self.lstm_value_output = []
        self.lstm_action_output = []
        self.rewards = 0
        for i in range(timesteps):
            print("current_frame_index", self.env.current_frame_index)
            self.dataset_train.append(t.tensor(obs, dtype=t.float32).permute(2, 0, 1).unsqueeze(0))

            if len(self.lstm_state_critic_h) == 0:
                action_logits, value, lstm_states_actor, lstm_states_critic, policy_features, value_features, x_value_rnn_output, x_action_rnn_output = self.wrapped_model(self.dataset_train[-1])
            else:
                action_logits, value, lstm_states_actor, lstm_states_critic, policy_features, value_features, x_value_rnn_output, x_action_rnn_output = self.wrapped_model(self.dataset_train[-1],lstm_states_actor, lstm_states_critic)


            self.values.append(value.detach().numpy()[0])
            self.lstm_states_actor_h.append(lstm_states_actor[0].detach().numpy())
            self.lstm_states_actor_c.append(lstm_states_actor[1].detach().numpy())
            self.lstm_state_critic_h.append(lstm_states_critic[0].detach().numpy())
            self.lstm_state_critic_c.append(lstm_states_critic[1].detach().numpy())
            self.lstm_value_output.append(x_value_rnn_output.detach().numpy())
            self.lstm_action_output.append(x_action_rnn_output.detach().numpy())
            self.mlp_policy_features.append(policy_features.detach().numpy())
            self.mlp_value_features.append(value_features.detach().numpy())

            action, states = self.model.predict(obs, state = states, deterministic=True)
            self.actions.append(action)
            self.lstm_hidden.append(states[0]) #combined policy and value
            self.lstm_cell.append(states[1]) #combined policy and value
            print("action", action)

            obs, reward, done, _, _ = self.env.step(action)
            self.rewards += reward
            print("reward",reward)

            if done:
                obs, _ = self.env.reset()
                print("total rewards",self.rewards)
                break


if __name__ == "main":
    TRAIN_VIDEO_PATH = "videos/V1-0001_City Scene Layout 1 setting0001.mp4"
    VIDEO_FPS = 24
    FRAME_SIZE = (124, 124)
    N_EPISODES = 10

    env = FrameGenerator(TRAIN_VIDEO_PATH, TRAIN_VIDEO_PATH, VIDEO_FPS, FRAME_SIZE,
                                         n_episodes=N_EPISODES, target_duration=4)
    model = RecurrentPPO.load('models/sb3_ppo_1goal_rnn_multistop_multiseq', env=env)

    wrapped_model = sb3Wrapper(model)

    target4_output_calculator = LayerOutputCalculator(wrapped_model,model,env)
    target4_output_calculator.get_layer_outputs_per_episode()


