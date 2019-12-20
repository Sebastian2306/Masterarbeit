import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

grid = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in grid:
        grid.append(k)
        
F0 = np.loadtxt('rwfoodF0.dat')
F1 = np.loadtxt('rwfoodF1.dat')
F2 = np.loadtxt('rwfoodF2.dat')
F5 = np.loadtxt('rwfoodF5.dat')

def func(t, alpha, y):
    return t**alpha * y


xdata = grid[10:]
ydata0 = F0[10:]
popt0, pcov0 = curve_fit(func, xdata, ydata0, bounds=(0, [1.0, 10]))
ydata1 = F1[10:]
popt1, pcov1 = curve_fit(func, xdata, ydata1, bounds=(0, [1.0, 10]))
ydata2 = F2[10:]
popt2, pcov2 = curve_fit(func, xdata, ydata2, bounds=(0, [1.0, 10]))
ydata5 = F5[10:]
popt5, pcov5 = curve_fit(func, xdata, ydata5, bounds=(0, [1.0, 10]))

plt.loglog(grid, F0, label = 'F0', lw = 0.5)
plt.loglog(grid, F1, label = 'F1', lw = 0.5)
plt.loglog(grid, F2, label = 'F2', lw = 0.5)
plt.loglog(grid, F5, label = 'F5', lw = 0.5)
plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 15)
plt.title('$20\%\ density$')
plt.legend()
plt.savefig('mult_food.pdf')

