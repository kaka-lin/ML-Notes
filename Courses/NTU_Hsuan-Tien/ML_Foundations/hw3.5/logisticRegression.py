import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sy # 代數運算，支援微分與積分

# define logistic function (sigmoid)
def Logistic(x):
	return 1.0 / (1 + np.exp(-x))
	# 補充：這邊不能使用math.exp()
	#  ->  only length-1 arrays can be converted to Python scalars
	# 因為math.exp()只能傳入長度為1的參數

###################################################################

# define batch gradient descent algorithm 
# 批量梯度下降演算法
def Gradient(w, x, y):
	# w_dim: (1 x d+1)
	# x_dim: (N x d+1) 
	# y_dim: (N, 1)

	# -ywx
	s = np.dot(x, w.transpose()) * -y 
	# θ(-ywx)
	theta = Logistic(s) 
	# θ(-ywx)(-yx)
	g = theta * (-y * x)
	return g.sum(axis = 0) / len(x)

	# 補充: NumPy中對 array 用 * or multiply()並不是矩陣乘法
	#      矩陣乘法：dot() 或 matmul()
	#      也可以將array類型轉乘matrix類型在使用 * (np.asmatrix())

def BathGD(T, ita, w, x, y):
	for i in range(T):
		w = w - ita * Gradient(w, x, y)
	return w

###################################################################

# define stochastic gradient descent algorithm
# 隨機梯度下降演算法
def Gradient2(w, x, y):
	s = np.dot(x, w.transpose()) * -y
	g = Logistic(s) * (-y * x)
	return g

def StochasticGD(T, ita, w, x, y, N):
	for i in range(T):
		w = w - ita * Gradient2(w, x[i % N], y[i % N])
	return w

###################################################################
def sign(w, x):
	if np.dot(w, x) >= 0:
		return 1
	else:
		return -1

# define Eout
def Eout(w, x, y):
	err = 0
	for i in range(len(x)):
		h = sign(w, x[i])
		if h != y[i]:
			err += 1
	return err / len(x)
###################################################################


if __name__ == "__main__":

	print("# Gradient Descent step?(default: η = 0.001)")
	ita = input("η: ")
	if ita == '':
		ita = 0.001
	else:
		ita = float(ita)
	#ita = 0.001 # ita = η

	print("# Iterative times?(default: T = 2000)")
	T = input("T: ")
	if T == '':
		T = 2000
	else:
		T = int(T)
	#T = 2000

	# train data
	train = np.loadtxt('hw3_train.dat')
	dim = len(train[0]) - 1 # dimension
	N = len(train)
	x = np.zeros((len(train), dim + 1))
	y = np.zeros((len(train), 1))  
	w_batch = np.zeros((1, dim + 1)) 
	w_stochastic = np.zeros((dim + 1))

	for i in range(len(train)):
		y[i] = train[i][dim]
		x[i] = np.append([1], np.delete(train[i], dim))

	#test data
	test = np.loadtxt('hw3_test.dat')
	dim_t = len(test[0]) - 1 # dimensio
	xt = np.zeros((len(test), dim_t + 1))
	yt = np.zeros((len(test), 1)) 

	for i in range(len(test)):
		yt[i] = test[i][dim_t]
		xt[i] = np.append([1], np.delete(test[i], dim_t))

	print()
	w_batch = BathGD(T, ita, w_batch, x, y)
	print("Batch Gradient Descent:\n", w_batch)
	print("Eout:", Eout(w_batch, xt, yt))

	print()
	w_stochastic = StochasticGD(T, ita, w_stochastic, x, y, N)	
	print("Stochastic Gradient Descent:\n", w_stochastic)
	print("Eout:", Eout(w_stochastic, xt, yt))
