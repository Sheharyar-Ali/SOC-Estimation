
# Model Explanation

This model uses a Kalman Filter to try and predict what the SOC of the car will be after n iterations. Most of the 
implementation is based on: https://www.mdpi.com/1996-1073/14/13/3733


## Cell model
An equivalent circuit model was used that is summarised here:
![Model](Images/Model.jpg)


This represents what each cell looks like

## State definitions

The state and output equations are as follow:

x_n+1 = A* x_n + B* u_n+1 + Q

y = C * x_n + D * u_n+1 + R

The states, x, used were: [SOC,I1,I2]

The input, u was [I_cell]

The output, y is the voltage across the cell, v

The matrices A and B were calculated based on the following equations:

![A+B](Images/A+B.png)

The matrix C was calculated based on:

![C](Images/C.png)

The following steps were then carried out:

1. First the matrices for the states and the state error are initialised in the matrices `xhat` and `Phat`
2. Then the states are predicted using the state equation. The results are stored in `xp`
3. Next, the state error and output are predicted and stored in `Pp` and `y` respectively
4. The Kalman gain, K, is calculated to allow the predictions to be corrected
5. `xp` and `Pp` are corrected and the resulting values are stored in `xc` and `Pc`
6. `xc` and `Pc` are used in the next iteration for Step 3.


The full equations for the steps are:

![Method](Images/Screenshot%20(55).png)

For information on the specific values chosen, refer to the `main.py` or `BatteryParams.py` file

# Usage 
Choose a simulation profile from `Extras/Simulation_profiles.py` to use in `main.py`. In `main.py`, define this function as the variable
`v_measured` and then run the code. Alternatively, in `BatteryParams.py` import the dataset you need

the outputs will be displayed in graphs once the code is done running. At the bottom of `main.py`, uncomment out any additional plots you want to see



