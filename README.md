# Deep Reinforcement Learning (DRL) video timer
This repository contains code to reproduce the results and analysis presented in the paper, **"Emergent time-keeping mechanisms in a deep reinforcement learning agent performing an interval timing task"**.



## Paper Overview
This paper investigates the emergent time-keeping mechanisms used by a deep reinforcement learning (DRL) agent while performing an interval timing task using visual inputs. 
We found that oscillatory patterns, a motif commonly observed in biological systems, can emerge as a time-keeping mechanism in artificial agents trained solely on an interval timing task. This study aims to initiate discussions about the similarities in time-keeping mechanisms that develop in artificial and biological systems through interactions with their environment.

You can read the full paper here: https://arxiv.org/pdf/2508.15784)

## 📂 Repository Structure

| Folder / File | Description                                                                                                                                                                                                                                                        |
| **`envs/`** | Contains the custom reinforcement learning environments. Each environment sends one video frame per timestep to the agent. |
| &nbsp;&nbsp;• `constant_frame_generation.py` | Repeats the same frame at each timestep. |
| &nbsp;&nbsp;• `frame_generator.py` | Uses consecutive frames from a coherent video sequence. |
| &nbsp;&nbsp;• `frame_generator_delayed.py` | Starts the interval timing task after a delay of *n* timesteps. |
| &nbsp;&nbsp;• `random_frame_generator.py` | Generates random frames at each timestep by shuffling a coherent video sequence. |

| **`train_agent.py`** | Main training script using a **CNN** for visual feature extraction, followed by an **LSTM**. Implements the `CnnLstmPolicy` policy of recurrent PPO from Stable Baselines3.                                                                                                                                                                                                       |
| **`train_agent_lstm.py`** | Alternative training setup using an **MLP** instead of a CNN. Implements the `MlpLstmPolicy` policy of recurrent PPO from Stable Baselines3.                                                                                                                                                                                                      |
| **`train_agent_delayed.py`** | Training script for the DRL agent on the **delayed interval timing task**. Can use CNN+LSTM or MLP+LSTM depending on the policy chosen. |
| **`videos/`** | Folder containing video inputs used by the environments. Any video can be used.                                                                                                                                                                                    |
                                                                                                                                                                                                                                   |
## Python Version

This code is tested with **Python 3.10**

## Key Features
- Trains DRL agents to perform interval timing tasks using video input.
- Multiple environment variants: delayed start, repeated frames, random frames.
- Analysis of internal oscillatory mechanisms for different target durations.
