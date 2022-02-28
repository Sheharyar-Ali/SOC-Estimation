1. we assume the discharge and charge curves are the same therefore there is no hysterisis
2. Thus coulombic efficieny is 100%
3. Matrix Q needs more information

#Model Explanation
This model uses a Kalman Filter to try and predict what the SOC of the car will be after n iterations. Most of the 
implementation is based on: https://www.mdpi.com/1996-1073/14/13/3733

The state and output equations are as follow:

x_n+1 = A* x_n + B* u_n+1 + Q

y = C * x_n + D * u_n+1 + R

The states, x, used were: [SOC,I1,I2]

The input, u was [I_cell]

The matrices A,B and C were calculated based on the following equations:


The following steps were then carried out:


For information on the specific values chosen, refer to the `main.py` or `BatteryParams.py` file

