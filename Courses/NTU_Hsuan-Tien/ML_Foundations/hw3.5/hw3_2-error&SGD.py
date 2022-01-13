import numpy as np
import matplotlib.pyplot as plt
import math

def sign(x):
	if x >= 0:
		return 1
	else:
		return -1

def logistic(x):
	return 1 / (1 + math.exp(-x))

def err01(x, y):
	if sign(x) == y:
		return 0
	else:
		return 1

def err1(x, y):
	return max(0, 1 - y * x)

def err2(x, y):
	return pow(max(0, 1 - y * x), 2)

def err3(x, y):
	return max(0, -y * x)

def err4(x, y):
	return logistic(-y * x)

def err5(x, y):
	return math.exp(-y * x)

x_range = np.arange(-2, 2, 0.0001)
y_log = []
err_01 = []
err_1 = []
err_2 = []
err_3 = []
err_4 = []
err_5 = []
y = 1

for i in range(len(x_range)):
	y_log.append(logistic(x_range[i]))
	err_01.append(err01(x_range[i], y))
	err_1.append(err1(x_range[i], y))
	err_2.append(err2(x_range[i], y))
	err_3.append(err3(x_range[i], y))
	err_4.append(err4(x_range[i], y))
	err_5.append(err5(x_range[i], y))

plt.figure(figsize = (20, 7))
plt.subplot(321)
plt.plot(x_range, y_log, label = r'$\frac{1}{1 + \mathrm{exp(-w^Tx)}}$', color = 'red' )
plt.plot(x_range, err_01, label = r'$err0/1$' )
plt.legend()

plt.subplot(322)
plt.plot(x_range, err_1, label = r'$(max(0, 1-y\mathrm{\mathbf{w^Tx}}))^2$', color = 'blue' ) # max(0, 1 − ywT x)
plt.plot(x_range, err_01, label = r'$[[sign(\mathrm{\mathbf{w^Tx}} )\neq y]]$' )
plt.legend()

plt.subplot(323)
plt.plot(x_range, err_2, label = r'$(max(0, 1-y\mathrm{\mathbf{w^Tx}}))^2$', color = 'black' ) #  pow(max(0, 1 − ywT x), 2)
plt.plot(x_range, err_01, label = r'$[[sign(\mathrm{\mathbf{w^Tx}} )\neq y]]$' )
plt.legend()

plt.subplot(324)
plt.plot(x_range, err_3, label = r'$max(0, -y\mathrm{\mathbf{w^Tx}})$', color = 'yellow' ) # max(0, −ywT x)
plt.plot(x_range, err_01, label = r'$[[sign(\mathrm{\mathbf{w^Tx}} )\neq y]]$' )
plt.legend()

plt.subplot(325)
plt.plot(x_range, err_4, label = r'$\theta(-y\mathrm{\mathbf{w^Tx}})$', color = 'green' ) # θ(−ywT x)
plt.plot(x_range, err_01, label = r'$[[sign(\mathrm{\mathbf{w^Tx}} )\neq y]]$' )
plt.legend()

plt.subplot(326)
plt.plot(x_range, err_5, label = r'$exp(-y\mathrm{\mathbf{w^Tx}})$', color = '#AE0000' ) # exp(−ywT x)
plt.plot(x_range, err_01, label = r'$[[sign(\mathrm{\mathbf{w^Tx}} )\neq y]]$' )
plt.legend()

plt.savefig('hw3_2.png', dpi = 100)
plt.show()
