import numpy as np
import torch as t

def get_extreme_weight_neurons(action_weights, threshold=2):

    # extreme weights
    action_weights_t = action_weights.T # (256,2)
    neuron_importance = t.norm(action_weights_t, dim=1).numpy() # get magnitude of weights across both actions, (256,)

    mean = np.mean(neuron_importance)
    std = np.std(neuron_importance)

    # Compute z-scores
    z_scores = (neuron_importance - mean) / std


    # Get indices of weights with extreme values
    high_z_indices = np.where(z_scores > threshold)[0]
    low_z_indices = np.where(z_scores < -threshold)[0]

    return high_z_indices, low_z_indices


def get_topn_neurons_per_action(action_weights, top_n=20):
    action_weights_t = action_weights.T
    topn_neurons = {}
    for i in range(action_weights_t.shape[1]):

        actioni_weights = action_weights_t[:, i]

        # Sort neurons by weights
        sorted_indices_actioni_weights = np.argsort(-actioni_weights)  # Descending order
        topn_neurons[i] = sorted_indices_actioni_weights[:top_n]
    return topn_neurons
