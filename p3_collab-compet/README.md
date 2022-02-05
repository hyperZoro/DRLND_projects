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
- Traing is done by code cell 5, average scores are printed per 100 episodes; the number of episodes required to reach the success criteria (score>=0.5) is also printed. The trained parameters are stored 4 _.pth_ files in the folder, for the Actors and Critics for each agent.
- Code cell 6 plots the trend of the score by the number of episode trained.
- Code cell 7 demonstrate the trained agent in action.
- The last code cell close the Unity enviroment.


### Results
I got the following training progress report in one of the attempts:
```
Episode 100:	Agents Mean Score over last 100 episodes: 0.00000 (time per eps:0.3 secs)
Episode 200:	Agents Mean Score over last 100 episodes: 0.00000 (time per eps:0.3 secs)
Episode 300:	Agents Mean Score over last 100 episodes: 0.00000 (time per eps:0.3 secs)
Episode 400:	Agents Mean Score over last 100 episodes: 0.00100 (time per eps:0.3 secs)
Episode 500:	Agents Mean Score over last 100 episodes: 0.00090 (time per eps:0.3 secs)
Episode 600:	Agents Mean Score over last 100 episodes: 0.00000 (time per eps:0.3 secs)
Episode 700:	Agents Mean Score over last 100 episodes: 0.00100 (time per eps:0.3 secs)
Episode 800:	Agents Mean Score over last 100 episodes: 0.01090 (time per eps:0.3 secs)
Episode 900:	Agents Mean Score over last 100 episodes: 0.01090 (time per eps:0.5 secs)
Episode 1000:	Agents Mean Score over last 100 episodes: 0.00480 (time per eps:0.3 secs)
Episode 1100:	Agents Mean Score over last 100 episodes: 0.02130 (time per eps:0.3 secs)
Episode 1200:	Agents Mean Score over last 100 episodes: 0.01480 (time per eps:1.0 secs)
Episode 1300:	Agents Mean Score over last 100 episodes: 0.01160 (time per eps:0.2 secs)
Episode 1400:	Agents Mean Score over last 100 episodes: 0.03230 (time per eps:0.6 secs)
Episode 1500:	Agents Mean Score over last 100 episodes: 0.02540 (time per eps:0.6 secs)
Episode 1600:	Agents Mean Score over last 100 episodes: 0.04530 (time per eps:0.3 secs)
Episode 1700:	Agents Mean Score over last 100 episodes: 0.05620 (time per eps:0.6 secs)
Episode 1800:	Agents Mean Score over last 100 episodes: 0.08860 (time per eps:0.6 secs)
Episode 1900:	Agents Mean Score over last 100 episodes: 0.10210 (time per eps:0.6 secs)
Episode 2000:	Agents Mean Score over last 100 episodes: 0.12660 (time per eps:1.3 secs)
Episode 2100:	Agents Mean Score over last 100 episodes: 0.13480 (time per eps:0.6 secs)
Episode 2200:	Agents Mean Score over last 100 episodes: 0.10900 (time per eps:0.6 secs)
Episode 2300:	Agents Mean Score over last 100 episodes: 0.17470 (time per eps:9.9 secs)
Episode 2400:	Agents Mean Score over last 100 episodes: 0.35300 (time per eps:6.8 secs)

Environment solved in 2450 episodes!	Average Score: 0.50210
```
The training process is not very consistent - sometimes it reaches the success criteria with a lot less episodes.

Score plot through episode count<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p3_collab-compet/pic/scores.png)

A trained agent in action.<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p3_collab-compet/pic/Animation.gif)
