import numpy as np
import scipy.integrate
from BatteryParams import e1, e2, Cap, T, dt, R0, R1, R2, C1, C2, sigma_i, R_internal_total, V_min, Current
from OCV_Calculation import OCV_25deg_og as OCV_60deg
from OCV_Calculation import SOC_OCV25deg
import matplotlib.pyplot as plt
from Extras.Simulation_profiles import V_Quadratic, V_linear

# states = [SOC,I1,I2]
# parameters = [R0,R1,R2,C1,C2,Discharge_Capacity]
# input = [I_cell]

# state equation:
# x_n+1 = A*x_n + B*u_n + Q
# output equation:
# y_n = OCV(SOC_n) - R0*I_cell_n - R1*I1_n - R2*I2_n


# Data needed 
v_measured = V_min  # choose the profile you want from Simulation profiles(like a pleb) or choose actual data
I = Current

#The entire KF has been made into a function
def KF(T, e1, e2, dt, Cap, R0, R1, R2, v_measured, I):

    #Setting arrays
    ycalc = np.zeros_like(T)
    Power_used = np.zeros_like(T)
    Energy_used = np.zeros_like(T)
    SOC = np.zeros_like(T)
    SOC_measured = np.zeros_like(T)
    I1 = np.zeros_like(T)
    I2 = np.zeros_like(T)
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
    # These values are based off estimates from: https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2560145

    u = I
    P = Phat


    for i in range(0, len(T)):
        print("Percentage completion:", i / len(T) * 100, "%")
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

        #This is just to make sure that the program does not crash when using the OVC-SOC relationship
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
        # SOC[i] = SOC_OCV25deg(y)
        I1[i] = xc[1][0]
        I2[i] = xc[2][0]
        P = Pc
        x = xc

        ycalc[i] = y
        SOC_measured[i] = SOC_OCV25deg(v_measured[i])
        Power_used[i] = y * I[i]
        if i != 0:
            Energy_calc = scipy.integrate.simpson(Power_used[0:i], T[0:i])
            Energy_used[i] = Energy_calc
    return ycalc, SOC, SOC_measured, Energy_used


ycalculated, SOC_calculated, SOC_v_min, Energy = KF(T=T, e1=e1, e2=e2, dt=dt, Cap=Cap, R0=R0,
                                                    R1=R1, R2=R2, v_measured=v_measured, I=I)
#
error_voltage = abs(v_measured - np.array(ycalculated))
avg_error_voltage = sum(error_voltage) / len(error_voltage)
print(avg_error_voltage)

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(T, v_measured, label="Measured")
axs[0].plot(T, ycalculated, label="Calculated")
axs[0].set_ylabel("Voltage [V]")
axs[0].set_ylim([3.3,4.2])
axs[0].legend()
axs[0].grid()

# axs[1].plot(T, error_voltage, label="error")
# axs[1].set_ylabel("absolute error in V")
# axs[1].plot(T, np.full_like(T, avg_error_voltage), label="avg")
# axs[1].legend()
# axs[1].grid()

axs[1].plot(T, SOC_v_min, label="SOC measured")
axs[1].plot(T, SOC_calculated, label="SOC calculated")
axs[1].set_ylabel("SOC")
axs[1].set_ylim([0,1])
axs[1].legend()
axs[1].grid()
# 
# axs[3].plot(T, error_SOC, label="error in SOC")
# axs[3].plot(T, np.full_like(T, avg_error_SOC), label="avg")
# axs[3].set_ylabel("Abs error in SOC")
# axs[3].legend()
# axs[3].grid()

axs[2].plot(T, Energy, label="Energy used")
axs[2].set_ylabel("Energy used [J]")
axs[2].legend()
axs[2].grid()
plt.show()
