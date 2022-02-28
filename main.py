import numpy as np
from numpy import random as rng
from BatteryParams import e1, e2, Cap, T, dt, R0, R1, R2, C1, C2, sigma_i, Tend
from OCV_Calculation import OCV_60deg_og as OCV_60deg
import matplotlib.pyplot as plt
from scipy import interpolate
from Extras.Simulation_profiles import V_Quadratic, V_linear

rng.seed(1)
# states = [SOC,I1,I2]
# parameters = [R0,R1,R2,C1,C2,Discharge_Capacity]
# input = [I_cell]

# state equation:
# x_n+1 = A*x_n + B*u_n + Q
# output equation:
# y_n = OCV(SOC_n) - R0*I_cell_n - R1*I1_n - R2*I2_n

# DATA ACQUISITION
SOC = np.zeros_like(T)
SOCp = []  # Used for the construction of the OCV-SOC lookup table
I1 = np.zeros_like(T)
I2 = np.zeros_like(T)
I_meas = np.zeros_like(T)
R = sigma_i ** 2  # measurement noise matrix
# This assumes that the only source of noise in the output equation will be the voltage sensor

# For Simulation purposes
v_measured = V_Quadratic(T)
I = np.zeros_like(T)
ycalc = []
# The Process noise matrix assumes that the noise is a result of the parameters used in matrices A and B
# Q = J*Qp*J^T
# Due to a lack of information, the Q matrix is assumed here instead of calculated

Q = np.diag([1E-4 ** 2,
             1E-3 ** 2,
             1E-6 ** 2])  # This is an estimation based on https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2560145

# Initialisation
xhat = np.array([[1.0],
                 [0.0],
                 [0.0]])  # state initialisation
Phat = np.diag([1, 1E-4 ** 2, 1E-4 ** 2])  # initial state error

u = I
P = Phat
# These values are based off estimates from: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2560145



for i in range(0, len(T)):
    print("STARTING ITERATION", i)
    if i == 0:
        x = xhat
        P = Phat

    # print("x", x)
    # The A,B, and C matrices for the state space system are calculated here
    A = np.diag([1, e1, e2])
    B = np.array([[-dt / (3600 * Cap)],
                  [(1 - e1)],
                  [(1 - e2)]])
    if i == 0:
        C = np.array([1, -R1, -R2])
    else:
        dOCV_dSOC = OCV_60deg(SOC[i-1])[1]
        C = np.array([dOCV_dSOC, -R1, -R2])
        I[i] = v_measured[i] * ((dt * OCV_60deg(SOC[i-1])[1]) - R0 - (1 - e1) - (1 - e2)) #This is for simulation process

    u = I[i]
    CT = np.atleast_2d(C).T  # This is needed to transpose a single row matrix in numpy
    # Prediction step
    xp = A @ x + B * u  # Predicting state
    if xp[0][0] < 0:
        xp[0][0] = 0
    elif xp[0][0] > 1:
        xp[0][0] = 1

    OCV_SOC_p = OCV_60deg(xp[0][0])[0]
    Pp = A @ P @ A.T + Q   # Predicting system state error
    #print("Pp", Pp)
    y = OCV_SOC_p - R0 * u - R1 * xp[1][0] - R2 * xp[2][0]  # Prediction of the output
    Denom = C @ Pp @ CT + R  # single value
    K = (Pp @ CT) * 1 / Denom  # Calculating Kalman gain
    # Correction step
    xc = xp + (K * (v_measured[i] - y))
    Pc = (np.eye(3) - (K * C)) @ Pp


    SOC[i] = xc[0][0]
    I1[i] = xc[1][0]
    I2[i] = xc[2][0]
    P = Pc
    x = xc

    # print(v_measured[i])
    # print(y)
    ycalc.append(y)

error = v_measured - np.array(ycalc)
fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(T[0:16000], v_measured[0:16000], label="Measured")
axs[0].plot(T[0:16000], ycalc[0:16000], label="Calculated")
axs[0].set_ylabel("Voltage [V}")
axs[0].legend()
axs[0].grid()

axs[1].plot(T[0:16000], error[0:16000], label="error")
axs[1].set_ylabel("error")
axs[1].legend()
axs[1].grid()

axs[2].plot(T[0:16000], SOC[0:16000], label="SOC calculated")
axs[2].set_ylabel("SOC")
axs[2].legend()
axs[2].grid()
plt.show()
