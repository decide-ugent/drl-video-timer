import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def plot_hidden_state(lstm_hidden, lstm_cell, high_z_indices, low_z_indices, top_neurons_a0, save_path=None):
    hidden_states_list = [h_t.flatten() for h_t in lstm_hidden]
    cell_states_list = [c_t.flatten() for c_t in lstm_cell]

    hidden_states_array = np.array(hidden_states_list)[:, :]
    cell_states_array = np.array(cell_states_list)[:, :]

    # filter for action 1 neurons
    high_z_indices = [i for i in high_z_indices if i not in top_neurons_a0]

    hidden_states_array_least_weight_neurons = np.array(hidden_states_list)[:, low_z_indices]  # remove 173
    # cell_states_array_least_weight_neurons = np.array(cell_states_list)[:,[8,205,125]]

    hidden_states_array_top_neurons = np.array(hidden_states_list)[:, high_z_indices]  # remove 39
    plt.figure(figsize=(14, 4))

    # Hidden state
    line_all = plt.plot(hidden_states_array, marker="o", alpha=0.2, linestyle="--", label="All neurons")
    line_least = plt.plot(hidden_states_array_least_weight_neurons, marker="o", color="red", linestyle="-",
                          label="Least weighted neurons")
    line_top = plt.plot(hidden_states_array_top_neurons, marker="o", color="blue", linestyle="-",
                        label="High weighted neurons")
    # plt.legend(handles=[line_all[0], line_least, line_top[0]])
    # plt.title("LSTM Hidden State Evolution Over Time")
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Hidden State Value", fontsize=12)


    plt.xticks(range(0, len(hidden_states_list)), fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def pca(squeezed_policy_layer, pca_components=3, total_timestep=20, save_path=None):
    scaler_policy = StandardScaler()

    policy_layer_scaled = scaler_policy.fit_transform(squeezed_policy_layer)

    pca_policy = PCA(n_components=pca_components)

    policy_layer_pc = pca_policy.fit_transform(policy_layer_scaled)

    explained_variance_policy = pca_policy.explained_variance_ratio_

    print("explained_variance policy", explained_variance_policy)

    colors = ['royalblue', 'deepskyblue', 'goldenrod']
    plt.figure(figsize=(8, 4))
    for i in range(policy_layer_pc.shape[1]):
        plt.plot(policy_layer_pc[:, i], marker='o', label=f'PC {i + 1} ({round(explained_variance_policy[i]*100)}%)', alpha=0.8, color=colors[i])

    # plt.title("Principal component analysis (PCA) of LSTM Hidden States Over Time", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    # plt.ylabel("Principal Component Value", fontsize=12)
    plt.xticks(range(policy_layer_pc.shape[0]))
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return policy_layer_pc


def fft(squeezed_policy_layer, high_z_indices, low_z_indices, save_path=None):
    # N - number of neurons
    # T - time steps
    squeezed_policy_layer_NT = squeezed_policy_layer.T  # Shape: (N, T)

    # Apply FFT to each neuron's time series
    fft_policy_results = np.fft.rfft(squeezed_policy_layer_NT, axis=1)
    fft_policy_magnitudes = np.abs(fft_policy_results)

    # Compute frequency bins
    time_steps = squeezed_policy_layer.shape[0]
    sampling_rate = 1  # 1 sample per timestep;
    freqs = np.fft.rfftfreq(time_steps, d=1 / sampling_rate)

    plt.figure(figsize=(8, 3))

    for i in range(squeezed_policy_layer.shape[1]):
        if i in high_z_indices:
            plt.plot(freqs[:time_steps ], fft_policy_magnitudes[i, :time_steps // 1], color="blue", marker="o")
        elif i in low_z_indices:
            plt.plot(freqs[:time_steps // 1], fft_policy_magnitudes[i, :time_steps // 1], color="red", marker="o")
        else:
            plt.plot(freqs[:time_steps // 1], fft_policy_magnitudes[i, :time_steps // 1], marker="o", alpha=0.05,
                     color="lightgrey", linestyle="--")

        # break

    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Magnitude", fontsize=12)
    plt.xticks(np.arange(0, 0.55, 0.05))
    plt.tight_layout()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
