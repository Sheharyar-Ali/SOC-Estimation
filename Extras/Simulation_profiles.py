import numpy as np
from BatteryParams import T,Tend
import matplotlib.pyplot as plt
from scipy import interpolate
maximum = 4.0
minimum = 3.3

def remap(old_value,old_min,old_max,new_max,new_min):
    new_value = (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min
    return new_value

xvals = np.array([0,1,2,3,4,5,6,7,8,10])
yvals = np.array([maximum,3.9,3.8,3.5,3.4,3.35,3.35,minimum,minimum,minimum])
print(len(xvals), len(yvals))
remapped_xvals = remap(xvals,0,10,max(T),0)

V_linear = maximum - ((maximum-minimum) / (Tend - 1)) * T
V_Quadratic = interpolate.interp1d(remapped_xvals,yvals,kind="quadratic")

#plt.plot(T,V_Quadratic(T))
#plt.show()