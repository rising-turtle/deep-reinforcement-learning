

#### Udacity Deep Reinforcement Learning Nanodegree
### Project 2: Continuous Control
# Train a Set of Robotic Arms

## Goal
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location (within the green spheres) for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

In order to solve the environment, our agent must achieve a score of +30 averaged across all 20 agents for 100 consecutive episodes.

##### &nbsp;

## Learning Algorithm 

The DDPG (Deep Deterministic Reinforcement Learning) is implemented, which consists of four neural networks: a Q network, a deterministic policy network, a target Q network, and a target policy network. The Q network and policy network is alike the Advantage Actor-Critic, but the Actor directly maps states to actions (the output of the network directly the output) instead of outputting the probability distribution across a discrete action space. The target networks are time-delayed copies of their original networks that iteratively updated through training in each episode. Using the target value networks stablizes the learning process.  

I used Adam for learning the neural network parameters with a learning rate of 2e−4 and 2e−3 for the actor and critic respectively. The L2 weight decay is 0 and the discount factor γ = 0.99. For the soft target updates τ = 0.001. The neural networks used the rectified non-linearity for two hidden layers. Also, a batch normalization is applied to the first hidden layer. The final output layer of the actor was a tanh layer, to bound the actions. Both the actor and critic networks had 2 hidden layers both with 128 units respectively. The final layer weights and biases of both the actor and critic were initialized from a uniform distribution [−3×10−3; 3×10−3]. This was to ensure the initial outputs for the policy and value estimates were near zero. 

The hyperparameters are shown below: 
```python
BUFFER_SIZE = int(1e5)           # replay buffer size
BATCH_SIZE = 128 #64 #128        # minibatch size
GAMMA = 0.99                     # discount factor
TAU = 1e-3                       # for soft update of target parameters
LR_ACTOR = 2e-4 #1e-4            # learning rate of the actor 
LR_CRITIC = 2e-4 # 1e-3          # learning rate of the critic
WEIGHT_DECAY = 0                 # L2 weight decay
LEARN_EVERY_N = 20               # learn every N intervals 
ACTOR_FC1_UNITS = 128 #400 # 100 # 400 
ACTOR_FC2_UNITS = 128 #300 # 100 #300 
CRITIC_FC1_UNITS = 128 #400 # 100 #400  
CRITIC_FC2_UNITS = 128 #300 # 100 #300  
```

The implementations of the Agent and the Actor-Critic are in the [ddpg_agent.py](ddpg_agent.py) and [model.py](model.py), respectively. Below is the initialization and forward functions of the Actor-Critic, demonstrating the structure of the networks: 

```python
class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=ACTOR_FC1_UNITS, fc2_units=ACTOR_FC2_UNITS):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=CRITIC_FC1_UNITS, fc2_units=CRITIC_FC2_UNITS):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

##### &nbsp;

## Exploration
For the exploration noise process we used temporally correlated noise in order to explore well in physical environments that have momentum. We used an Ornstein-Uhlenbeck process with θ = 0:15 and σ = 0.1. The Ornstein-Uhlenbeck process models the velocity of a Brownian particle with friction, which results in temporally correlated values centered around
0.

##### &nbsp;

## Results

The graph below shows the training process, showing that the moving average score is  at th episode. The complete set of results and steps can be found in [this notebook](Continuous_Control.ipynb).

<img src="assets/results-graph.png" width="70%" align="top-left" alt="" title="Results Graph" />

<img src="assets/output.png" width="100%" align="top-left" alt="" title="Final output" />


##### &nbsp;

## Future Improvements
- I am out of element here, maybe try the STOA D4PG algorithm [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617).

##### &nbsp;
##### &nbsp;

---



