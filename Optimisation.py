from main import T, dt, Cap, OCV_60deg, e1, e2, R0, R1, R2, R_internal_total, sigma_i,V_min,Current
import numpy as np
from OCV_Calculation import OCV_60deg_og as OCV_60deg
from OCV_Calculation import SOC_OCV60deg
import matplotlib.pyplot as plt
from Extras.Simulation_profiles import V_Quadratic, V_linear
# DATA ACQUISITION
SOC = np.zeros_like(T)
SOC_measured = np.zeros_like(T)
I1 = np.zeros_like(T)
I2 = np.zeros_like(T)

# For Simulation purposes
v_measured = V_min  # choose the profile you want from Simulation profiles(like a pleb) or choose actual data
I = Current
I_calc = v_measured / R_internal_total
ycalc = np.zeros_like(T)

# Noise settings
# The Process noise matrix assumes that the noise is a result of the parameters used in matrices A and B
# Q = J*Qp*J^T
# Due to a lack of information, the Q matrix is assumed here instead of calculated

Q = np.diag([1E-4 ** 2,
             1E-3 ** 2,
             1E-6 ** 2])  # This is an estimation based on https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2560145

R = sigma_i ** 2  # measurement noise matrix
# This assumes that the only source of noise in the output equation will be the voltage sensor

# Initialisation
xhat = np.array([[1.0],
                 [0.0],
                 [0.0]])  # state initialisation
Phat = np.diag([1, 1E-4 ** 2, 1E-4 ** 2])  # initial state error

u = I
P = Phat


# These values are based off estimates from: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2560145

def KF_optimisation(e1, T, xhat, Phat, e2, dt, Cap, R0, R1, R2, v_measured, I, SOC, SOC_measured, ycalc):
    for i in range(0, len(T)):
        # print("Percentage completeion:", i / len(T) * 100, "%")
        if i == 0:
            x = xhat
            P = Phat

        # Matrix computation
        # The A and B matrices for the state space system are calculated here
        A = np.diag([1, e1, e2])
        B = np.array([[-dt / (3600 * Cap)],
                      [(1 - e1)],
                      [(1 - e2)]])

        if i == 0:
            C = np.array([1, -R1, -R2])
        else:
            # print(SOC[i-1])
            dOCV_dSOC = OCV_60deg(SOC[i - 1])[1]
            C = np.array([dOCV_dSOC, -R1, -R2])
            # I_calc[i] = v_measured[i] * (
            #        (dt * OCV_60deg(SOC[i - 1])[1]) - R0 - (1 - e1) - (1 - e2))  # This calculates the current based
            # on the voltage for simulation purposes

        u = I[i]
        CT = np.atleast_2d(C).T  # This is needed to transpose a single row matrix in numpy

        # Prediction step
        xp = A @ x + B * u  # Predicting state

        if xp[0][0] < 0:
            xp[0][0] = 0
        elif xp[0][0] > 1:
            xp[0][0] = 1

        OCV_SOC_p = OCV_60deg(xp[0][0])[0]
        Pp = A @ P @ A.T + Q  # Predicting system state error
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

        ycalc[i] = y
        SOC_measured[i] = SOC_OCV60deg(v_measured[i])
    return ycalc, SOC, SOC_measured


error_list = []
avg_error_list = []
data_fraction = 1
fraction = int(data_fraction * len(T) - 1)
T = T[0:fraction]
v_measured = v_measured[0:fraction]
I = I[0:fraction]
ycalc = ycalc[0:fraction]

# Optimisation ranges
R0_values = np.linspace(0, 0.002, 10)
R1_values = np.linspace(0, 0.01, 10)
C1_values = np.linspace(1E3, 1E6, 10)
C2_values = np.linspace(1E3, 1E8, 5)
e1_values = np.exp(-dt / (R1 * C1_values))
e2_values = np.exp(-dt / (R2 * C2_values))
for i in range(0, len(C2_values)):
    e2 = e2_values[i]
    print("completion:", (i / len(C2_values)) * 100, C2_values[i])
    calc, SOC, SOC_v_measured = KF_optimisation(e1=e1, T=T, xhat=xhat, Phat=Phat, e2=e2, dt=dt, Cap=Cap, R0=R0, R1=R1,
                                                R2=R2,
                                                v_measured=v_measured, I=I, SOC=SOC, SOC_measured=SOC_measured,
                                                ycalc=ycalc)

    error = abs(calc - v_measured)
    error_list.append(error)
    avg_error_list.append(sum(error) / len(error))
    if avg_error_list[i] == min(avg_error_list):
        min_err = avg_error_list[i]
        min_err_index = i

plt.plot(C2_values, avg_error_list, marker="o")
print("minimum error", min_err, "at index", min_err_index)
print("e1 minimum", C2_values[min_err_index])
print(avg_error_list)
plt.legend()
plt.show()
