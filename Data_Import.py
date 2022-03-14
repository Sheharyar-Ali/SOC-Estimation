import numpy as np
import matplotlib.pyplot as plt
import csv
from OCV_Calculation import plotter

file = open("/home/sheharyar/SOC-Estimation/DUT19_FSA.csv") #the full path is needed to run the Optimisation file
type(file)

reader = csv.reader(file)

header = next(reader)

rows =[]
Time = []
V_min = []
Current = []
for row in reader:
    rows.append(row)
    #print(row)
    if 0.5 <= float(row[0]) and row[0] !='' and row[1]!='':
        Time.append(float(row[0])) #filters out the time needed to boot up
        V_min.append(float(row[1]))
        Current.append((float(row[2]) / 1000))
Time = np.array(Time)
V_min=np.array(V_min)
Current = np.array(Current)/2

