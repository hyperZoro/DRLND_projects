# DRLND - Solution to Project 2: Continuous Control
Udacity Deep Reinforcement Learning Nano Degree Project 2 Solution

This folder contains the solution to Project 2: Continuous Control, as part of the Udacity Deep Reinforcement Learning Nano Degree.

[Project rubric](https://review.udacity.com/#!/rubrics/1890/view)
## About the project
The goal of the project is to train the agent(s) to control double jointed arm(s), to track the target locations.
### The reward dynamic and the goal of the agent
Each double-jointed arm can move to target locations, A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal is to maintain its positiion at the target location to maximise total reward.

### State space
The state of the enviroment is described discretely by 33 state values (a 1D-vector of size 33), correspondingto position, rotation, velocity, and angular velocities of the arm(s).
### Action space
4 continuous variables (a 1D-vector of size 4) describe the action space for each agent. The values corresponds to torque applicable to the two joints. Every entry in the action vector is a number between -1 and 1. 
### Termination of the episode
The task is episodic, it is terminated at the maximum number of time step (1001), when the accumuated reward is counted as the final reward. 
### Success criteria
The sucess criteria is achieving an average score of +30 over 100 consecutive episodes.

## The Python enviroment
The enviroment is prepared as instructed by [this page](https://github.com/udacity/deep-reinforcement-learning#dependencies). The solution is developed in Windows 11.
I got a list of Python package dependence from the Peer Chat Channel, which is recorded in _requirement.txt_. mujoco, as part of the OpenAI Gym dependence, failed to build. PyTorch v0.4 is replaced by 1.10.1, and intel-openmp v2020.0.133 is replaced by v2022.0.0, both due to availability from the default repo. 
I drop the Windows 64-bit version Unity Enviroment in folder: _./Reacher_Windows_x86_64_1/_, which I got from below.
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

The _ipynb_ file has a path string pointing to the executable.

## Deep Deterministic Policy Gradient (DDPG)
The solution is based on the DDPG example codes provided in the ddpg folder of [this repo](https://github.com/udacity/deep-reinforcement-learning). There is a very nice description provided [here](https://spinningup.openai.com/en/latest/algorithms/ddpg.html).

The following files show the solution:
- ddpg_agent.py
- ddpg_model.py
- Report.ipynb

_Report.ipynb_ contains the write-up of the report; while the trained parameters are saved in _checkpoint_actor_ddpg.pth_ and checkpoint_critic_ddpg.

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
Agents Mean Score in Episode 005: 1.17, 	Agents Mean Score over last 100 episodes: 0.59 (time per eps:55.4 secs)
Agents Mean Score in Episode 010: 2.62, 	Agents Mean Score over last 100 episodes: 1.13 (time per eps:56.3 secs)
Agents Mean Score in Episode 015: 5.20, 	Agents Mean Score over last 100 episodes: 2.30 (time per eps:55.4 secs)
Agents Mean Score in Episode 020: 8.58, 	Agents Mean Score over last 100 episodes: 3.47 (time per eps:56.1 secs)
Agents Mean Score in Episode 025: 13.17, 	Agents Mean Score over last 100 episodes: 4.94 (time per eps:58.3 secs)
Agents Mean Score in Episode 030: 19.58, 	Agents Mean Score over last 100 episodes: 7.00 (time per eps:57.6 secs)
Agents Mean Score in Episode 035: 29.16, 	Agents Mean Score over last 100 episodes: 9.86 (time per eps:59.6 secs)
Agents Mean Score in Episode 040: 32.64, 	Agents Mean Score over last 100 episodes: 12.53 (time per eps:60.7 secs)
Agents Mean Score in Episode 045: 35.19, 	Agents Mean Score over last 100 episodes: 14.86 (time per eps:61.9 secs)
Agents Mean Score in Episode 050: 36.58, 	Agents Mean Score over last 100 episodes: 16.90 (time per eps:63.9 secs)
Agents Mean Score in Episode 055: 38.22, 	Agents Mean Score over last 100 episodes: 18.76 (time per eps:63.1 secs)
Agents Mean Score in Episode 060: 38.27, 	Agents Mean Score over last 100 episodes: 20.35 (time per eps:63.4 secs)
Agents Mean Score in Episode 065: 38.45, 	Agents Mean Score over last 100 episodes: 21.72 (time per eps:63.6 secs)
Agents Mean Score in Episode 070: 38.57, 	Agents Mean Score over last 100 episodes: 22.91 (time per eps:63.0 secs)
Agents Mean Score in Episode 075: 37.82, 	Agents Mean Score over last 100 episodes: 23.95 (time per eps:63.6 secs)
Agents Mean Score in Episode 080: 38.41, 	Agents Mean Score over last 100 episodes: 24.85 (time per eps:63.1 secs)
Agents Mean Score in Episode 085: 37.18, 	Agents Mean Score over last 100 episodes: 25.58 (time per eps:62.9 secs)
Agents Mean Score in Episode 090: 38.04, 	Agents Mean Score over last 100 episodes: 26.24 (time per eps:63.5 secs)
Agents Mean Score in Episode 095: 38.61, 	Agents Mean Score over last 100 episodes: 26.87 (time per eps:62.7 secs)
Agents Mean Score in Episode 100: 37.41, 	Agents Mean Score over last 100 episodes: 27.42 (time per eps:63.0 secs)
Agents Mean Score in Episode 105: 37.18, 	Agents Mean Score over last 100 episodes: 29.29 (time per eps:63.2 secs)

Environment solved in 107 episodes!	Average Score: 30.02
```
Score plot through episode count. First plot is scores for all individual agents; second plot is for the averaged score per episode (blue) and per 100 consecutive episodes (red)<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p2_continuous-control/pic/scores.png)

20 trained agents in action.<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p2_continuous-control/pic/Animation.gif)
