import numpy as np
import matplotlib.pyplot as plt
import csv
from OCV_Calculation import plotter

file = open("Endurance_Data.csv")
type(file)

reader = csv.reader(file)

header = next(reader)

rows =[]
Time = []
V_min = []
for row in reader:
    rows.append(row)
    if float(row[0])>= 0.5:
        Time.append(float(row[0])) #filters out the time needed to boot up
        V_min.append(float(row[1]))
Time = np.array(Time)
V_min=np.array(V_min)
plotter(Time,V_min,V_min)

