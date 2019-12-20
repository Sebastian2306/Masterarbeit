import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

grid = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in grid:
        grid.append(k)
        
s10 = np.loadtxt('suc10.dat')
s20 = np.loadtxt('suc20.dat')


def func(t, alpha, y):
    return t**alpha * y


xdata = grid[10:]
ydata0 = s10[10:]
popt0, pcov0 = curve_fit(func, xdata, ydata0, bounds=(0, [1.0, 10]))
ydata1 = s20[10:]
popt1, pcov1 = curve_fit(func, xdata, ydata1, bounds=(0, [1.0, 10]))


plt.loglog(grid, s10, label = '$F=5\ at\ 10\%\ density$', lw = 0.5)
plt.loglog(grid, grid, label = '$reference$', lw = 0.5)
plt.loglog(grid, s20, label = '$F=5\ at\ 20\%\ density$', lw = 0.5)
plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 15)
plt.title('$successiv\ algorithm$')
plt.legend()
plt.savefig('suc.pdf')

