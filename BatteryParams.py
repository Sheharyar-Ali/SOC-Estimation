import numpy as np
from Data_Import import Time as T

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
dt = T[1]-T[0]
Tend = T[-1]


R_internal_total = 4.2e-3  # assumption based on:
R1 = 0.001 #0.2 * R_internal_total
C1 = 1E3
R2 = 0.005 #0.3 * R_internal_total/home/sheharyar/SOC-Estimation
C2 = 11112000.0
R0 = 0.001 #0.5 * R_internal_total
Cap = 6.55  # [Ah] assumed to be same as nominal capacity


e1 = np.exp(-dt / (R1 * C1))
e2 = np.exp(-dt / (R2 * C2))

sigma_i = 0.5  # current sensor precision
