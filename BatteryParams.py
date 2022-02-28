import numpy as np
from scipy import interpolate, integrate
from OCV_Calculation import OCV_60deg, ynew, xvals

# %% LiPo cell transient model (linear)

# electrical model for real LiPo cell:
#
#  +--------------------- [LOAD] ------------------+
#  |                                               |
#  |                                               |
#  -                                               |
# (V)  <-- ideal cell                              |
#  +                                               |
#  |               +--- R1 ---+    +--- R2 ---+    |
#  |               |          |    |          |    |
#  +------- R0 ----+ fast RC  +----+ slow RC  +----+
#                  |          |    |          |
#                  +--- C1 ---+    +--- C2 ---+
#
#          ^ -------- losses ------ ^
#
dt = 100e-3
Tend = 1600
T = np.arange(0, Tend, dt)

R_internal_total = 4.2e-3  # assumption based on:
R1 = 0.2 * R_internal_total
C1 = 300
R2 = 0.3 * R_internal_total
C2 = 60e3
R0 = 0.5 * R_internal_total
Cap = 6.55  # [Ah] assumed to be same as nominal capacity

params = [R0, R1, R2, C1, C2, Cap]

e1 = np.exp(-dt / (R1 * C1))
e2 = np.exp(-dt / (R2 * C2))

sigma_i = 0.5  # current sensor precision
