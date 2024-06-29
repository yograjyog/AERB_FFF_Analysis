import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Load user data file
root = Tk()
root.withdraw()  # Hide the root window
profile = filedialog.askopenfile()

# Define column names and load data into a dataframe
user_cols = ['Distance', 'Dose']
df1 = pd.read_table(profile, names=user_cols, header=None)

# Define first and second derivative functions using central finite difference method
h = df1.iloc[1]['Distance'] - df1.iloc[0]['Distance']
X = df1['Distance']
Y = df1['Dose']


def f1(i):
    return (df1.iloc[i + 1]['Dose'] - df1.iloc[i - 1]['Dose']) / (2 * h)


def f2(i):
    return (df1.iloc[i + 2]['Dose'] - 2 * df1.iloc[i]['Dose'] + df1.iloc[i - 2]['Dose']) / (4 * h * h)


# Calculate first and second derivatives
X1 = df1.iloc[1:len(X) - 1, 0]
X2 = df1.iloc[2:len(X) - 2, 0]
Y1 = [f1(i) for i in range(1, len(Y) - 1)]
Y2 = [f2(i) for i in range(2, len(Y) - 2)]

df2 = pd.DataFrame(list(zip(Y1, Y2)), columns=["f'(x)", 'f"(x)'])
Data = pd.concat([df1.iloc[1:len(X) - 1].reset_index(drop=True), df2], axis=1)

# Finding the Beam Profile parameters
# Inflection Points
Rt_maxslope = int(Y1.index(max(Y1)))
Rt_ipvalues = Y2[Rt_maxslope - 6:Rt_maxslope + 4]
Rt_disvalue = X[Rt_maxslope - 4:Rt_maxslope + 6]
Rt_dosevalue = list(Y[Rt_maxslope - 4:Rt_maxslope + 6])
RtSign_Change = int(np.where(np.diff(np.sign(Rt_ipvalues)))[0])

Lt_minslope = int(Y1.index(min(Y1)))
Lt_ipvalues = Y2[Lt_minslope - 5:Lt_minslope + 5]
Lt_disvalue = X[Lt_minslope - 3:Lt_minslope + 7]
Lt_dosevalue = list(Y[Lt_minslope - 3:Lt_minslope + 7])
LtSign_Change = int(np.where(np.diff(np.sign(Lt_ipvalues)))[0])

RtIP_df = pd.DataFrame(list(zip(Rt_disvalue, Rt_dosevalue, Rt_ipvalues)), columns=["Rt_dis", "Rt_dose", "f''(x)"])
LtIP_df = pd.DataFrame(list(zip(Lt_disvalue, Lt_dosevalue, Lt_ipvalues)), columns=["Lt_dis", "Lt_dose", "f''(x)"])

Rt_RDV = (RtIP_df.loc[RtSign_Change, "Rt_dose"] + RtIP_df.loc[RtSign_Change + 1, "Rt_dose"]) / 2
Lt_RDV = (LtIP_df.loc[LtSign_Change, "Lt_dose"] + LtIP_df.loc[LtSign_Change + 1, "Lt_dose"]) / 2
RDV = (Rt_RDV + Lt_RDV) / 2

# Field Size
Rt_Fwidth = (RtIP_df.loc[RtSign_Change, "Rt_dis"] + RtIP_df.loc[RtSign_Change + 1, "Rt_dis"]) / 2
Lt_Fwidth = (LtIP_df.loc[LtSign_Change, "Lt_dis"] + LtIP_df.loc[LtSign_Change + 1, "Lt_dis"]) / 2
Field_Width = Lt_Fwidth - Rt_Fwidth
Field_Size = Field_Width / 10

# Beam Penumbra
ZeroPoint = int(df1[df1["Distance"] == 0].index.values)
Rt_X = X[:ZeroPoint + 1]
Rt_Y = Y[:ZeroPoint + 1]
Rt_df = pd.DataFrame(list(zip(Rt_X, Rt_Y)), columns=["Rt_X", "Rt_Y"])

Rt_Pa = 1.6 * Rt_RDV
Rt_Pb = 0.4 * Rt_RDV
Lt_X = X[ZeroPoint:]
Lt_Y = Y[ZeroPoint:]
Lt_df = pd.DataFrame(list(zip(Lt_X, Lt_Y)), columns=["Lt_X", "Lt_Y"])

Lt_Pa = 1.6 * Lt_RDV
Lt_Pb = 0.4 * Lt_RDV
pa = pb = None

for value in Rt_Y:
    if value >= Rt_Pa:
        pa = Rt_df[Rt_df["Rt_Y"] == value].index.values.astype(int)[0]
        break
for value in Rt_Y:
    if value >= Rt_Pb:
        pb = Rt_df[Rt_df["Rt_Y"] == value].index.values.astype(int)[0]
        break

pa_Rt = (Rt_df.loc[pa, "Rt_X"] + Rt_df.loc[pa - 1, "Rt_X"]) / 2
pb_Rt = (Rt_df.loc[pb, "Rt_X"] + Rt_df.loc[pb - 1, "Rt_X"]) / 2
Rt_penumbra = (pb_Rt - pa_Rt)

for value in Lt_Y:
    if value <= Lt_Pa:
        pa = Lt_df[Lt_df["Lt_Y"] == value].index.values.astype(int)[0]
        break
for value in Lt_Y:
    if value <= Lt_Pb:
        pb = Lt_df[Lt_df["Lt_Y"] == value].index.values.astype(int)[0]
        break

pa_Lt = (Lt_df.loc[pa - 1, "Lt_X"] + Lt_df.loc[pa, "Lt_X"]) / 2
pb_Lt = (Lt_df.loc[pb - 1, "Lt_X"] + Lt_df.loc[pb, "Lt_X"]) / 2
Lt_penumbra = (pb_Lt - pa_Lt)

# Horizontal distances for 90%, 75%, and 60% dose levels
dose_levels = [90, 75, 60]
horizontal_distances = {}


# Linear interpolation function
def linear_interpolate(x1, y1, x2, y2, y):
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


# Function to find the distances at a specific dose value on both sides
def find_distances_at_dose(dose_value):
    left_distance = None
    right_distance = None

    for i in range(len(Y) - 1):
        if (Y[i] <= dose_value <= Y[i + 1]) or (Y[i + 1] <= dose_value <= Y[i]):
            interpolated_distance = linear_interpolate(X[i], Y[i], X[i + 1], Y[i + 1], dose_value)
            if X[i] < 0 and left_distance is None:
                left_distance = interpolated_distance
            elif X[i] > 0 and right_distance is None:
                right_distance = interpolated_distance

    return left_distance, right_distance


# Find the distances at 90%, 75%, and 60% dose values on both sides
left_distance_90, right_distance_90 = find_distances_at_dose(90)
left_distance_75, right_distance_75 = find_distances_at_dose(75)
left_distance_60, right_distance_60 = find_distances_at_dose(60)

# Calculate the horizontal distances between the right and left dose points for each dose level
horizontal_distance_90 = abs(right_distance_90 - left_distance_90)
horizontal_distance_75 = abs(right_distance_75 - left_distance_75)
horizontal_distance_60 = abs(right_distance_60 - left_distance_60)

# Output the calculated values
print(Data)
print("Right IP =", Rt_RDV)
print("Left IP =", Lt_RDV)
print("Average RDV =", RDV)
print("Field Size =", Field_Size, "cm")
print("Rt Penumbra =", Rt_penumbra, "mm")
print("Lt Penumbra =", Lt_penumbra, "mm")
print("Left distance at 90% dose =", left_distance_90, "mm")
print("Right distance at 90% dose =", right_distance_90, "mm")
print("Horizontal distance between right and left 90% dose points =", horizontal_distance_90, "mm")
print("Left distance at 75% dose =", left_distance_75, "mm")
print("Right distance at 75% dose =", right_distance_75, "mm")
print("Horizontal distance between right and left 75% dose points =", horizontal_distance_75, "mm")
print("Left distance at 60% dose =", left_distance_60, "mm")
print("Right distance at 60% dose =", right_distance_60, "mm")
print("Horizontal distance between right and left 60% dose points =", horizontal_distance_60, "mm")

# Plotting Graph
fig = plt.figure()
graph = fig.add_subplot(1, 1, 1)
graph.spines['left'].set_position(('data', 0.0))
graph.set_xlim(-150, 150)
graph.set_ylim(-5, 100)
major_xticks = np.arange(-150, 150, 10)
major_yticks = np.arange(-5, 105, 5)
minor_xticks = np.arange(-150, 150, 1)
minor_yticks = np.arange(-5, 105, 1)
graph.set_xticks(major_xticks)
graph.set_xticks(minor_xticks, minor=True)
graph.set_yticks(major_yticks)
graph.set_yticks(minor_yticks, minor=True)
graph.grid(True, which='major', color='white', linestyle='-')
graph.grid(True, which='minor', color='white', linestyle='-')
plt.plot(X, Y, label="Beam Profile", color="blue", linewidth=0.5, marker="*", ms=1)

# Annotate distances on the plot
plt.axvline(x=left_distance_90, color='green', linestyle='--', label='Left 90% Dose Distance')
plt.axvline(x=right_distance_90, color='orange', linestyle='--', label='Right 90% Dose Distance')
plt.text(left_distance_90, 50, f'Left 90% = {left_distance_90:.2f} mm', rotation=90, verticalalignment='center')
plt.text(right_distance_90, 50, f'Right 90% = {right_distance_90:.2f} mm', rotation=90, verticalalignment='center')

plt.axvline(x=left_distance_75, color='purple', linestyle='--', label='Left 75% Dose Distance')
plt.axvline(x=right_distance_75, color='brown', linestyle='--', label='Right 75% Dose Distance')
plt.text(left_distance_75, 40, f'Left 75% = {left_distance_75:.2f} mm', rotation=90, verticalalignment='center')
plt.text(right_distance_75, 40, f'Right 75% = {right_distance_75:.2f} mm', rotation=90, verticalalignment='center')

plt.axvline(x=left_distance_60, color='blue', linestyle='--', label='Left 60% Dose Distance')
plt.axvline(x=right_distance_60, color='red', linestyle='--', label='Right 60% Dose Distance')
plt.text(left_distance_60, 30, f'Left 60% = {left_distance_60:.2f} mm', rotation=90, verticalalignment='center')
plt.text(right_distance_60, 30, f'Right 60% = {right_distance_60:.2f} mm', rotation=90, verticalalignment='center')

plt.text((left_distance_90 + right_distance_90) / 2, 90, f'Distance 90% = {horizontal_distance_90:.2f} mm',
         horizontalalignment='center')
plt.text((left_distance_75 + right_distance_75) / 2, 75, f'Distance 75% = {horizontal_distance_75:.2f} mm',
         horizontalalignment='center')
plt.text((left_distance_60 + right_distance_60) / 2, 60, f'Distance 60% = {horizontal_distance_60:.2f} mm',
         horizontalalignment='center')

plt.xlabel("Distance from central axis (cm)")
plt.ylabel("% Relative Dose")
plt.title(profile.name)
plt.legend()
plt.show()
