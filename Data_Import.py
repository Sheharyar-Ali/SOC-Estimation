import numpy as np
import matplotlib.pyplot as plt
import csv
from OCV_Calculation import plotter

file_1 = open("/home/sheharyar/SOC-Estimation/DUT19_FSA.csv")  # the full path is needed to run the Optimisation file
file_2 = open("/home/sheharyar/SOC-Estimation/DUT19_Vmin+current.csv")
type(file_1)
type(file_2)

reader_FSA = csv.reader(file_1)
reader_Endurance = csv.reader(file_2)

header_FSA = next(reader_FSA)
header_Endurance = next(reader_Endurance)

# Importing FSA data
rows_FSA = []
Time_FSA = []
V_min_FSA = []
Current_FSA = []
for row in reader_FSA:
    rows_FSA.append(row)
    # print(row)
    if 0.5 <= float(row[0]) and row[0] != '' and row[1] != '':
        Time_FSA.append(float(row[0]))  # filters out the time needed to boot up
        V_min_FSA.append(float(row[1]))
        Current_FSA.append((float(row[2]) / 1000))
Time_FSA = np.array(Time_FSA)
V_min_FSA = np.array(V_min_FSA)
Current_FSA = np.array(Current_FSA) / 2

# Importing endurance data
rows_Endurance = []
Time_Endurance = []
V_min_Endurance = []
Current_Endurance = []
for row in reader_Endurance:
    rows_Endurance.append(row)
    # print(row)
    if 0.5 <= float(row[0]) and row[0] != '' and row[1] != '':
        Time_Endurance.append(float(row[0]))  # filters out the time needed to boot up
        V_min_Endurance.append(float(row[1]))
        Current_Endurance.append((float(row[2]) / 1000))
Time_Endurance = np.array(Time_Endurance)
V_min_Endurance = np.array(V_min_Endurance)
Current_Endurance = np.array(Current_Endurance) / 2

# plt.plot(Time_FSA, Current_FSA, label="FSA")
# plt.plot(Time_Endurance, Current_Endurance, label="Endurance")
# plt.grid()
# plt.legend()
# plt.show()
