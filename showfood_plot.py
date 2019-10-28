import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

grid = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in grid:
        grid.append(k)
        
rw10 = np.loadtxt('rw10.dat')
rw20 = np.loadtxt('rw20.dat')
rw50 = np.loadtxt('rw50.dat')

food10 = np.loadtxt('food10.dat')
food20 = np.loadtxt('food20.dat')
food50 = np.loadtxt('food50.dat')


def func(t, alpha, y):
    return t**alpha * y


xdata = grid[10:]
ydata0 = rw10[10:]
popt0, pcov0 = curve_fit(func, xdata, ydata0, bounds=(0, [1.0, 10]))
ydata1 = rw20[10:]
popt1, pcov1 = curve_fit(func, xdata, ydata1, bounds=(0, [1.0, 10]))
ydata2 = rw50[10:]
popt2, pcov2 = curve_fit(func, xdata, ydata2, bounds=(0, [1.0, 10]))

plt.subplot(2,1,1)
plt.loglog(grid, grid, 'b', label = '$reference$', lw = 0.5)
plt.loglog(grid, rw10, 'g', label = '$F=5\ at\ 10\%\ density$', lw = 0.5)
plt.loglog(grid, rw20, 'm', label = '$F=5\ at\ 20\%\ density$', lw = 0.5)
plt.loglog(grid, rw50, 'r', label = '$F=5\ at\ 50\%\ density$', lw = 0.5)
plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 15)
plt.title('$normal\ algorithm$')
plt.legend()
plt.subplot(2,1,2)
plt.plot(grid, food10, 'g')
plt.plot(grid, food20, 'm')
plt.plot(grid, food50, 'r')
plt.xscale('log')
plt.xlabel('$steps$', size = 15)
plt.ylabel('$food$', size = 15)
plt.savefig('showfood.pdf')


