# DRLND - Solution to Project 3: Collaboration and Competition
Udacity Deep Reinforcement Learning Nano Degree Project 3 Solution

This folder contains the solution to Project 3: Collaboration and Competition, as part of the Udacity Deep Reinforcement Learning Nano Degree.

[Project rubric](https://review.udacity.com/#!/rubrics/1891/view)

## Building the Python enviroment
The enviroment is prepared as instructed by [this page](https://github.com/udacity/deep-reinforcement-learning#dependencies). The solution is developed in Windows 11.
I got a list of Python package dependence from the Peer Chat Channel, which is recorded in _requirement.txt_. mujoco, as part of the OpenAI Gym dependence, failed to build. PyTorch v0.4 is replaced by 1.10.1, and intel-openmp v2020.0.133 is replaced by v2022.0.0, both due to availability from the default repo. 

I dropped the Windows 64-bit version Unity Enviroment in folder: _./Tennis_Windows_x86_64/_, which I got from below.
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

The _ipynb_ file has a path string pointing to the executable.


## About the enviroment and the agent
The goal of the project is to train two agents to control two rackets to play tennis, i.e. rally the ball over the net. 

### The reward dynamic and the goal of the agent
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

### State space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each variable is represented as a vector of 3 valuess. Therefore, the state received by each agent is described by the 1-D vector of 24 values.
```
Vector Observation space type: continuous
Vector Observation space size (per agent): 8
Number of stacked Vector Observation: 3
```

### Action space
Two continuous actions are available for each racket, corresponding to movement toward (or away from) the net, and jumping. The action values should be between -1 and +1, represented by a vector of 2 values.

### Termination of the episode
The task is episodic. When the episode is terminated, the accumuated score is counted for each agent, and the max score between the two agent is reported as the final score. 

### Success criteria
The agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


## Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
The solution is based on the MADDPG approach described in [thie paper](https://arxiv.org/abs/1706.02275).

The following files show the solution:
- ddpg_agent.py
- ddpg_model.py
- Report.ipynb

_Report.ipynb_ contains the write-up of the report; while the trained parameters are saved in _checkpoint_actor_ddpg.pth_ and checkpoint_critic_ddpg. One can follow the steps below to run the code cells in _Report.ipynb_ to replicate what I did.

### Running the code
- Set up the Python enviroment according to "The Python enviroment".
- Download the Coninuous Control Unity enviroment.
- Update the path pointing to the Unity enviroment executable file in the second code cell in _Report.ipynb_, where variable _env_ is set.
- Run the code cells one by one from top to bottom in _Report.ipynb_. 
- Traing is done by code cell 5, average scores are printed per 5 episodes; the number of episodes required to reach the success criteria (score>=30) is also printed. The trained parameters are stored in _checkpoint_actor_ddpg.pth_ and _checkpoint_actor_ddpg.pth_.
- Code cell 7 plots the trend of the score by the number of episode trained.
- Code cell 8 demonstrate the trained agent in action.
- The last code cell close the Unity enviroment.


### Results
I got the following training progress report:
```

```
Score plot through episode count<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p2_continuous-control/pic/scores.png)

A trained agent in action.<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p2_continuous-control/pic/Animation.gif)
