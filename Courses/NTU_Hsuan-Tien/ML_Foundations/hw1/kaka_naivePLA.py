# Perceptron Hypothesis
# h(x) = sign(W^T*x)

# Perceptron Learning Algorithm
# if sign(w^T*x) != y
# => w(t+1) = w(t)+y(t)x(t)
'''舉個例子：某次學習錯誤將原本結果該為 −1的 yn(t)誤判成 +1
這代表 w 與 x 之間的角度太小，所以我們透過 wt+1←wt+(−1)xn(t)
來將夾角加大，使未來學習到這點時能正確的判斷成 −1
'''

'''
Each line of the data set contains one (xn,yn) with xn ∈ R4. 
The first 4 numbers of the line contains 
the components of xn orderly, the last number is yn. 
Please initialize your algorithm with w = 0 and take sign(0) as −1. 
As a friendly reminder, remember to add x0 = 1 as always!

#15
Implement a version of PLA by visiting examples in the naıve cycle 
using the order of examples in the data set. 
Run the algorithm on the data set. 
What is the number of updates before the algorithm halts? 
What is the index of the example that results in the most number of updates?
'''

import pprint
import numpy as np

# load data
datas = np.loadtxt('hw1_15_train.dat')  
'''
for row in datas:
	print(row)
'''	

# creat x, y, w
# because add x0 = 1 -> xn ∈ R5
x = np.zeros((len(datas), 5))
y = np.zeros((len(datas), 1))
w = np.zeros((1, 5))

for i in range(len(datas)):
	y[i] = datas[i][4]
	x[i] = np.append([1], np.delete(datas[i],4))
	'''
	x[i][0] = datas[i][0]
	x[i][1] = datas[i][1]
	x[i][2] = datas[i][2]
	x[i][3] = datas[i][3]
	'''
	
# define Perceptron Hypothesis	
def h(w, x):
	if np.dot(w, x) > 0:
		return 1
	else:
		return -1

# Perceptron Learning Algorithm(Naive PLA)
# navi cycle的意思是:
# 每次發生錯誤時，循環不重新開始，而是繼續跑下去 
iteration = 0
naive_times = 0
update_index = dict()
while True:
	flag = 0
	for i in range(0, len(datas)):
	    g = h(w, x[i])
	    if g != y[i]:
	    	w = w + y[i] * x[i]
	    	iteration += 1
	    	flag = 1
	    	if i in update_index:
	    		update_index[i] += 1
	    	else:
	        	update_index[i] = 1
	if flag == 0:
	    break
	else:
	    naive_times += 1  

# What is the index of the example 
# that results in the most number of updates?
max = 1
max_index = dict()
for k, v in update_index.items():
    if v > max:
    	max = v
for k, v in update_index.items():
	if v == max:
		max_index[k] = v

print(w)
print(iteration)
print(naive_times)
pprint.pprint(max_index)
# pprint.pprint(update_index)