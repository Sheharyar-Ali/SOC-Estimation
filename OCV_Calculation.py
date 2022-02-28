from scipy import interpolate
import numpy as np
from matplotlib import pyplot as plt


# The functions below are two ways of recreating the OCV-SOC curve and its derivatives. The first function seems to work better
def OCV_60deg_og(SOC):
    Voltage_60deg = [3.30, 3.76, 3.8, 3.82, 3.89, 3.91, 3.93, 3.95, 3.98, 4, 4.07, 4.2]
    Voltage_60deg = np.array(Voltage_60deg)
    SOC_c = [0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.75, 0.9, 1]
    SOC_c = np.array(SOC_c)

    OCV_60deg = interpolate.interp1d(SOC_c, Voltage_60deg, kind="cubic")
    differential_function = interpolate.InterpolatedUnivariateSpline(SOC_c, Voltage_60deg)
    differential = differential_function.derivative()

    return OCV_60deg(SOC), differential(SOC)


def OCV_60deg(SOC):
    discharge_linear = np.array([0, 2.3, 30, 70])
    Voltage_linear = np.array([4.2, 4.11, 4.0, 3.87])
    discharge_final = np.array([70, 80, 90, 100])
    Voltage_final = np.array([3.87, 3.82, 3.76, 3.3])

    SOC_linear = np.array([0.3, 0.7, 0.977, 1])  # Need SOC not discharge for the differential
    Voltage_linear_ordered = np.array([3.87, 4.0, 4.11, 4.2])
    SOC_nonlinear = np.array([0, 0.1, 0.2, 0.3])  # Need SOC not discharge for the differential
    Voltage_nonlinear_ordered = np.array([3.3, 3.76, 3.82, 3.87])

    OCV_linear = interpolate.interp1d(discharge_linear, Voltage_linear, kind="linear")
    OCV_nonlinear = interpolate.interp1d(discharge_final, Voltage_final, kind="cubic")

    OCV_SOC_linear = interpolate.interp1d(SOC_linear, Voltage_linear_ordered, kind="linear")
    OCV_SOC_nonlinear = interpolate.InterpolatedUnivariateSpline(SOC_nonlinear, Voltage_nonlinear_ordered)

    discharge_wanted = (1 - SOC) * 100
    if discharge_wanted < 0:
        discharge_wanted = 0

    if discharge_wanted <= 70:
        OCV = OCV_linear(discharge_wanted)
    else:
        OCV = OCV_nonlinear(discharge_wanted)

    # calculating differential
    if SOC >= 0.3 and SOC < 0.977:
        derivative = (OCV_SOC_linear(SOC) - OCV_SOC_linear(0.3)) / (SOC - 0.3)
    elif SOC > 0.977:
        derivative = (OCV_SOC_linear(1) - OCV_SOC_linear(SOC)) / (1 - SOC)
    else:
        derivative = OCV_SOC_nonlinear.derivative()(SOC)

    return OCV, derivative


def plotter(xvals, yvals, yder):
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(xvals, yvals)
    axs[0].grid()

    axs[1].plot(xvals, yder)

    plt.show()


xvals = np.linspace(0, 1, 1000)

ynew = np.zeros_like(xvals)
yder = np.zeros_like(xvals)
for i in range(0, len(ynew)):
    ynew[i] = OCV_60deg_og(xvals[i])[0]
    yder[i] = OCV_60deg_og(xvals[i])[1]

# plotter(xvals,ynew,yder) #Uncomment this to plot the graph of OCV-SoC
