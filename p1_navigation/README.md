# DRLND - Solution to Project 1: Banana Navigation
Udacity Deep Reinforcement Learning Nano Degree Project 1 Solution

This folder contains the solution to Project 1: Navigation, as part of the Udacity Deep Reinforcement Learning Nano Degree.
## Briefly about enviromen
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
- Navigation.ipynb

where I basically adopted the new enviroment state and action counts as well as API, and ran it as is. It is a success though. The notebook _Navigation.ipynb_ has some more detailed descriptions on the model and agent, and trained parameters are saved in _checkpoint.pth_.

I got the following scores:
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
- Navigation_plus.ipynb

It is not quite a success (yet), mainly due to unbearable speed (6 hours and only completed 100 episode). I suspect it is to do with some mis-use ot for-loops or the way how I update values in certain objects. The Notebook _Navigation_plus.ipynb_ has some more detailed write-ups. In particular, a PER section describe the intension of the modification and the problems and suspections.


