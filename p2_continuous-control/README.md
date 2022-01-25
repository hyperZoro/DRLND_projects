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
- Download the Banana Navigation Unity enviroment.
- Update the path pointing to the Unity enviroment executable file in the second code cell in _Report.ipynb_, where variable _env_ is set.
- Run the code cells one by one from top to bottom in _Report.ipynb_. Note code cell 4 and 5 are optional, just to test the enviroment and agent are imported correctly.
- Traing is done by code cell 6, average scores are printed per 100 episodes; the number of episodes required to reach the success criteria (score>=13) is also printed. The trained parameters are stored in _checkpoint.pth_.
- Code cell 7 plots the trend of the score by the number of episode trained.
- Code cell 8 demonstrate the trained agent in action.
- The last code cell close the Unity enviroment.


### Results
When I used a stricter success criteria of score>=16, I got the following progress report:
```
Episode 100	Average Score: 1.17
Episode 200	Average Score: 4.36
Episode 300	Average Score: 7.39
Episode 400	Average Score: 9.16
Episode 500	Average Score: 13.02
Episode 600	Average Score: 14.30
Episode 700	Average Score: 14.35
Episode 800	Average Score: 15.06
Episode 900	Average Score: 14.78
Episode 993	Average Score: 16.00
Environment solved in 893 episodes!	Average Score: 16.00
```
Score plot through episode count<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p1_navigation/pic/score.png)

A trained agent in action, this one actually gets a score of **21**<br />
![](https://github.com/hyperZoro/DRLND_projects/blob/main/p1_navigation/pic/Animation1.gif)
