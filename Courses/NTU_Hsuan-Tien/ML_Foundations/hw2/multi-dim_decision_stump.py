import numpy as np
import matplotlib.pyplot as plt

# def sign(x)
def sign(x):
	if x >= 0:
		return 1
	else:
		return -1
# def h(x) = s*sing(x-theta)
def h(s, theta, x):
	return s * sign(x - theta)
# ====================== train ====================== #
#load data
datas = np.loadtxt('hw2_train.dat')

n = len(datas[0]) - 1 # x dimension
x = np.zeros((len(datas), n))
y = np.zeros((len(datas), 1))
new_x = np.zeros((n, len(datas)))
new_y = np.zeros((n, len(datas)))

for i in range(len(datas)):
	y[i] = datas[i][9]
	x[i] = np.append([], np.delete(datas[i],n))


for dim in range(n):
	single_data = np.column_stack((x[:,dim],y))
	# argsort函數返回的是陣列值從小到大的索引值
	sort_index = np.argsort(single_data, axis = 0)
	sort_data = single_data[np.transpose(sort_index)[0]]
	new_x[dim] = np.transpose(sort_data)[0]
	new_y[dim] = np.transpose(sort_data)[1]
	'''
	補充: np.argsort()
	a = np.array([[10, 2], [9, 1], [8, 3]])
	b = np.argsort(a, axis=0)

	    [10 2] -> [2 1]
		[9  1]    [1 0]
		[8  3]    [0 2]

	a = a[b] -> 此為通過索引值排序後的陣列
	    [2 1] -> [[[8  3]
	    		   [9  1]]
	    [1 0]	  [[9  1]
	    		   [10 2]] 
		[0 2]     [[10 2]
		           [8  3]]]
    np.transpose(b) -> [2 1 0]
                       [1 0 2]
    => a = a[np.transpose(b)[0]] 即可得到以第一行排序後的陣列
	'''

best_err = len(x)
best_s = 0
best_theta = 0
best_dimension = 0

for i in range(n):
	# 將資料兩端加上端點，[-1+x0, 1+xn] -> 比x0還小的點與比xn還大的點    
    x1 = np.append(np.append([-1 + new_x[i][0]], new_x[i]),[1 + new_x[i][len(new_x[i])-1]])
    # 計算theta : 兩點之中值
    theta = [(x1[j] + x1[j+1]) / 2 for j in range(len(new_x[i]) + 1)] # List Comprehensions
    
    one_best_err = len(x)
    one_best_s = 0
    one_best_theta = 1

    for k in range(len(theta)):
    	err = 0 # s = 1
    	err_nag = 0 # s = -1
    	for z in range(len(new_x[i])):
    		g = h(1, theta[k], new_x[i][z])
    		g_nag = h(-1, theta[k], new_x[i][z])
    		if g != new_y[i][z]:
    			err += 1
    		if g_nag != new_y[i][z]:
    			err_nag += 1
    	if err < err_nag:
    		if err < one_best_err:
    			one_best_err = err
    			one_best_s = 1
    			one_best_theta = theta[k]
    	else:
    		if err_nag < one_best_err:
    			one_best_err = err_nag
    			one_best_s = -1
    			one_best_theta = theta[k]

    if one_best_err < best_err:
    	best_err = one_best_err
    	best_dimension = i + 1
    	best_s = one_best_s    	
    	best_theta = one_best_theta

print("index of the dimension where the optimal decision stump is generated is", best_dimension)
print("Ein = ", best_err / len(x),", s = ", best_s, ", theta = ", best_theta)

# ====================== test ====================== #
test = np.loadtxt("hw2_test.dat")

n_test = len(test[0]) - 1 # x dimension
x_test = np.zeros((len(test), n))
y_test = np.zeros((len(test), 1))
new_x_test = np.zeros((n, len(test)))
new_y_test = np.zeros((n, len(test)))

for i in range(len(test)):
	y_test[i] = test[i][9]
	x_test[i] = np.append([], np.delete(test[i],n))


for dim in range(n_test):
	single_data = np.column_stack((x_test[:,dim], y_test))
	# argsort函數返回的是陣列值從小到大的索引值
	sort_index = np.argsort(single_data, axis = 0)
	sort_data = single_data[np.transpose(sort_index)[0]]
	new_x_test[dim] = np.transpose(sort_data)[0]
	new_y_test[dim] = np.transpose(sort_data)[1]

Eout = 0
for i in range(len(new_x_test[best_dimension])):
	if best_s == 1:
		g = h(1, best_theta, new_x_test[best_dimension - 1][i])
		if g != new_y_test[best_dimension - 1][i]:
			Eout += 1
	if best_s == -1:
		g = h(-1, best_theta, new_x_test[best_dimension - 1][i])
		if g != new_y_test[best_dimension - 1][i]:
			Eout += 1
print("Eout = ", Eout / len(x_test))
