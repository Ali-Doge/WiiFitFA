from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

data = np.loadtxt('./data/Ali/Ali_walk_1.csv', delimiter=',', dtype=str)
data = data[1:, :]
data = data.astype(int)

emg_data = data[:, 0:8]
acc_data = data[:, 12:15]
gyr_data = data[:, 15:18]
data_arrs = [emg_data, acc_data, gyr_data]


fig, subplots = plt.subplots(8, 3)

name = "tab10" # Change this if you have sensors > 10
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list

for j, axis in enumerate(data_arrs):
    for i in range(axis.shape[1]):
        subplots[i,j].plot(axis[:, i], color=colors[i])

plt.show()
