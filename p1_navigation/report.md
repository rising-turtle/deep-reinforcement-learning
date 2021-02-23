[img_scores_plot]: Capture.PNG

Report Project 1: Navigation 

Banana Collector:
The environment is a modified version of the Banana Collector Environment from the Unity ML-Agents toolkit. I adopted the Deep Q-learning algorithm to train a banana collector agent. The implementation consists of two classes and one function which are detailed below: 

1. QNetwork in model.py: The Q-network architecture has three hidden fully connected layers (fc1: 64x1, fc2: 64x1, fc3: 16x1, with ReLU activations) and one output layer (with no activation). The input to our network is the agent's state representation (37x1 output of the preprocess step φ), and the output layer outputs 4 q-values for each possible action in the environment.

2. Agent in dqn_agent.py: The agent applied the experience replay strategy to reduce the correlation between sequential training samples. It uses TD-target as the estimate of the true q-values that the Q-Network seeks to output given a specific pair (state, action). The TD-target js a separate Q-Network to compute the target q values, which has the same architecture as the network trained for taking actions. The difference is that the weights of the target Q-Network are copied from the weights of the training Q-network (local) after some specific number of steps. 

3. function dqn() in Navigation.ipynb: It runs maximum 1000 episodes to train the network. In each episode, the maximum steps are 1000. The scores and the latest 100 scores are stored, and if the mean score for the lastest 100 episodes is above 13, meaning that this problem has been solved. Then, it stops and saves the trained network to 'model1.pt'. The training scores verse episode are plotted below: 

![q-learning-process][img_scores_plot]


