#### READ ME #### 
#########################################################
ACROBOT CONTROL SYSTEM 
#########################################################

#########################################################
FILES STRUCTURE
#########################################################

Main Files: 

- main.py - main execution scripts, coordinate all tasks
- parameters.py  configuration file for system parameters

Dynamics: 

- dynamics.py - Double pendulum dynamics 
- Test_Dynamics.py - Test dynamic model with zero input
- task1.py : Equilibrium computation and reference trajectory generation

Optimization and Control: 

newton_optcon.py - Newton optimization with Riccati backward pass and Armijo line search
Newton_Loop.py - Main Newton loop with plotting functions
cost.py - Cost function definitions (stage cost and terminal cost)
LQR.py - Linear Quadratic Regulator for tracking control
MPC.py - Model Predictive Control 

Others: 

animation.py - pendulum animation visualization

#########################################################
HOW TO USE
#########################################################

In parameters.py modify the tasks_to_run dictionary:

tasks_to_run = {
    0: True,  # Test dynamics with zero input
    1: True,  # Newton optimization - step reference
    2: True,   # Newton optimization - smooth reference
    3: True,   # LQR tracking control
    4: True,   # MPC tracking control
    5: True    # Animations
}

Set to True the tasks you want to execute. 
IMPORTANT: You need to execute at least one time previous tasks to run next tasks

Run the Code: 

RUN main.py

The code will execute selected tasks sequentially and save results in the data/ folder.
