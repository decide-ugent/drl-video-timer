import torch.nn as nn


class sb3Wrapper(nn.Module):
    def __init__(self, model):
        super(sb3Wrapper,self).__init__()
        # self.features_extractor_cnn = model.policy.features_extractor.cnn
        # self.features_extractor_linear = model.policy.features_extractor.linear
        self.pi_features_extractor = model.policy.pi_features_extractor
        self.vf_features_extractor = model.policy.vf_features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net
        self.lstm_actor = model.policy.lstm_actor
        self.lstm_critic = model.policy.lstm_critic

    def forward(self,x, x_value_rnn=None,x_action_rnn=None, deterministic=True ):
        # print(x.shape)
        x_pi = self.pi_features_extractor(x)
        x_vf = self.vf_features_extractor(x)
        # x = self.mlp_extractor(x)
        # print(x.shape)
        x_value_rnn_output, x_value_rnn = self.lstm_critic(x_vf, x_value_rnn)
        x_action_rnn_output, x_action_rnn = self.lstm_actor(x_pi, x_action_rnn)
        x_value = self.value_net(x_value_rnn_output)
        x_action = self.action_net(x_action_rnn_output)
        # distribution = model.policy._get_action_dist_from_latent(x_action_rnn_output)
        # actions = distribution.get_actions(deterministic=deterministic)
        # log_prob = distribution.log_prob(actions)

        return x_action, x_value, x_action_rnn, x_value_rnn, x_pi, x_vf, x_value_rnn_output, x_action_rnn_output