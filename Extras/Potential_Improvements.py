import numpy as np
from BatteryParams import e1, e2, Cap, T, R0, R1, R2, C1, C2, sigma_i
from OCV_Calculation import OCV_60deg
import matplotlib.pyplot as plt
from scipy import interpolate

## This file does not work but rather contains the calculations and matrices needed for a better prediction of the Q matrix


params = [R0, R1, R2, C1, C2, Cap] # The parameters that dictate the noise matrix
# The Process noise matrix assumes that the noise is a result of the parameters used in matrices A and B
# Q = J*Qp*J^T
# The differentials for the Jacobian are calculated here:
dSOC_dCap = (T * I) / (3600 * Cap ** 2)
dI1_dR1 = (-I1 / R1) * e1 + (I / R1) * e1
dI1_dC1 = (-I1 / C1) * e1 + (I / C1) * e1
dI2_dR2 = (-I2 / R2) * e2 + (I / R2) * e2
dI2_dC2 = (-I2 / C2) * e2 + (I / C2) * e2

# The qp matrix is given here:
Qp = []  # We need the std deviations of our parameters here

J = np.array([[0, 0, 0, 0, 0, dSOC_dCap[i]],
              [0, dI1_dR1[i], 0, dI1_dC1[i], 0, 0],
              [0, 0, dI2_dR2[i], 0, dI2_dC2[i], 0]])  # The Jacobian for the Process noise calculation