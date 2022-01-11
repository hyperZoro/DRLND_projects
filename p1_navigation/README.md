# DRLND - Solution to Project 1: Banana Navigation
Udacity Deep Reinforcement Learning Nano Degree Project 1 Solution

This folder contains the solution to Project 1: Navigation, as part of the Udacity Deep Reinforcement Learning Nano Degree.
## About the project
The goal of the project is to train an agent to navigate the enviroment, where it collect bananas within a room.
### The reward dynamic and the goal of the agent
There are yellow and violet bananas randomly located within the room, collecting one yellow banana will generate +1 reward; while collecting one violet banana generate -1 reward. The goal of the agent is to collect as many yellow bananas as possible (highest positive reward) within the timeframe.
### State space
The state of the enviroment is described discretely by 37 state values (a 1D-vector of size 37).
### Action space
One of 4 possible **discrete** actions (move forward, backward, left or right) can be chosen by the agent at each time step.
### Termination of the episode
The task is episodic, it is terminated at the maximum number of time step (300), when the accumuated reward is counted as the final reward. 
### Success criteria
The sucess criteria is achieving higher than an average acculuated reward of 13 over 100 episodes.

## The Python enviroment
The enviroment is prepared as instructed by [this page](https://github.com/udacity/deep-reinforcement-learning#dependencies). The solution is developed in Windows 11.
I got a list of Python package dependence from the Peer Chat Channel, which is recorded in _requirement.txt_. mujoco, as part of the OpenAI Gym dependence, failed to build. PyTorch v0.4 is replaced by 1.10.1, and intel-openmp v2020.0.133 is replaced by v2022.0.0, both due to availability from the default repo. 
I drop the Windows 64-bit version of the Banana Navigation Unity Enviroment in folder: _./Banana_Windows_x86_64/_, which I got from below. The _ipynb_ file has a path string pointing to the executable.
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## First attempt
As a first attempt, I simply copy the solution from the Deep Q-Network section. The following three files show this minimal effort
- dqn_agent.py
- model.py
- Report.ipynb

where I basically adopted the new enviroment state and action counts as well as API, and ran it as is. It is a success though. The notebook _Navigation.ipynb_ has some more detailed descriptions on the model and agent, and trained parameters are saved in _checkpoint.pth_.

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

## Attempt with prioritised experience replay
Based on the first attempt, I have tried to add the Prioritised Experience Replay feature. The following files are relevant.

- dqn_agent_plus.py
- model.py
- Report_plus.ipynb

It is not quite a success (yet), mainly due to unbearable speed (6 hours and only completed 100 episode). I suspect it is to do with some mis-use ot for-loops or the way how I update values in certain objects. The Notebook _Navigation_plus.ipynb_ has some more detailed write-ups. In particular, a PER section describe the intension of the modification and the problems and suspections.


