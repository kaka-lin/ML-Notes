import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x_data = [ 338.,  333.,  328. , 207. , 226.  , 25. , 179. ,  60. , 208.,  606.]
y_data = [  640.  , 633. ,  619.  , 393.  , 428. ,   27.  , 193.  ,  66. ,  226. , 1591.]

x = np.arange(-200, -100, 1) # bias
y = np.arange(-5, 5, 0.1) # weight
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y) # 把向量擴展成矩陣

for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# ydata = b + w * xdata
b = -120 # initial b
w = -4 # initial w
lr = 1 # learning rate
iteration = 100000

b_grad_last = 0
w_grad_last = 0

# Store initial values for plotting
b_history = [b]
w_history = [w]

x_data, y_data = np.array(x_data), np.array(y_data)


# Iterations
for i in range(iteration):
    b_grad = -2.0 * (y_data - b - w * x_data)
    b_grad = np.sum(b_grad) / len(b_grad)
    w_grad =  np.dot(x_data.transpose(), -2.0 * (y_data - b - w * x_data))
    #print('{}. b_grad:'.format(i), b_grad, ', {}. w_grad:'.format(i), w_grad)
    
    b_grad_last = b_grad_last + b_grad ** 2
    w_grad_last = w_grad_last + w_grad ** 2
    #print('{}. b_grad_last:'.format(i), b_grad_last, ', {}. w_grad_last:'.format(i), w_grad_last)
    
    # Update parameters(Use Adagrad)
    b = b - lr/np.sqrt(b_grad_last) * b_grad
    w = w - lr/np.sqrt(w_grad_last) * w_grad

    # Store parameters for plotting
    b_history.append(b)
    w_history.append(w)


# plot the figure
plt.contourf(x,y,Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
