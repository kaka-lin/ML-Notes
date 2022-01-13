# from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import random  
import pprint

# define Perceptron Hypothesis	
def h(w, x):
	if np.dot(w, x) > 0:
		return 1
	else:
		return -1

# load data
datas = np.loadtxt('hw1_15_train.dat')  

x = np.zeros((len(datas), 5))
y = np.zeros((len(datas), 1))
w = np.zeros((1, 5)) 

for i in range(len(datas)):
	y[i] = datas[i][4]
	x[i] = np.append([1], np.delete(datas[i],4))
		
'''
Perceptron Learning Algorithm(random_cycle PLA)
random cycle的意思是: 每次實驗前把X的順序打亂
然後重覆2000次
'''

# 先產生放置2000次結果之陣列
repeat = 2000
result = np.zeros((1, repeat))

for k in range(repeat):
	# 產生random data
	random_sort = range(len(x))
	# random.sample -> 從參數1中隨機選取參數2個元素
	random_index = random.sample(random_sort, len(x))

	#每次開始時w都需歸零
	w = np.zeros((1, 5)) 

	iteration = 0
	naive_times = 0
	update_index = dict()

	while True:
		flag = 0
		for i in range(len(datas)):
			j = random_index[i]
			g = h(w, x[j])
			if g != y[j]:
				w = w + y[j] * x[j]
				iteration += 1
				flag = 1
		if flag == 0:
			break
		else:
			naive_times += 1  

	result[0][k] = iteration

pprint.pprint(result)
print(np.average(result))

bins = range(100)
plt.hist(result[0], bins, histtype='bar', rwidth=0.8)
plt.xlabel("Update times")
plt.ylabel("Frequency")
plt.show()