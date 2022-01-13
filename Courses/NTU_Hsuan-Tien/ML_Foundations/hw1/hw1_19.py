# from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import random  
import pprint
import copy

# define Perceptron Hypothesis	
def h(w, x):
	if np.dot(w, x) > 0:
		return 1
	else:
		return -1

# load train data
datas = np.loadtxt("hw1_18_train.dat")  
x = np.zeros((len(datas), 5))
y = np.zeros((len(datas), 1))
for i in range(len(datas)):
	x[i] = np.append([1], np.delete(datas[i],4))
	y[i] = datas[i][4]

# load test data 
test_datas = np.loadtxt("hw1_18_test.dat")
xt = np.zeros((len(test_datas), 5))
yt = np.zeros((len(test_datas), 1))
for i in range(len(test_datas)):
    xt[i] = np.append([1], np.delete(test_datas[i],4))
    yt[i] = test_datas[i][4]  

# 先產生放置2000次結果之陣列
repeat = 2000
result = np.zeros((1, repeat))

for times in range(repeat):
	# 產生random data
	random_sort = range(len(x))
	# random.sample -> 從參數1中隨機選取參數2個元素
	random_index = random.sample(random_sort, len(x))

	# 每次開始時w都需歸零
	w = np.zeros((1, 5))
	wn = np.zeros((1, 5)) # w with least error rate in train data so far

	# w update 次數
	update = 0

	# least error rate
	least_error = 1 * len(datas) # 先假設w在資料上全錯
	for i in range(len(datas)):
		if (update == 100): # 只更新100次
			break
		j = random_index[i]
		g = h(w, x[j])
		if g != y[j]:
			# w += y[j] * x[j] -> python少用+=會出包
			# 因為他是生成一個新物件在綁定過去
			# 原本的值變垃圾還是在那邊，容易誤導
			w = w + y[j] * x[j]
			update += 1
						
			error = 0
			for k in range(len(datas)):
				g_err = h(w, x[k])
				if g_err != y[k]:
					error += 1

			if error < least_error:
				least_error = error
				# 將值賦值給另一個變數時
				# 最好使用copy
				# 如果使用參照容易被原本變垃圾的物件影響到
				wn = copy.deepcopy(w) 
	error_test = 0            
	for i in range(len(test_datas)):
		g_test = h(wn, xt[i])
		if g_test != yt[i]:
			error_test += 1	
	result[0][times] = (error_test / len(test_datas)) * 100

print("The average error rate in test data is: ", np.average(result[0]),"%")
bins = range(100)
plt.hist(result[0], bins, histtype='bar', rwidth=0.8)
plt.xlabel("Error rate (%)")
plt.ylabel("Frequency")
plt.savefig('hw1_19.png')
#plt.savefig('hw1_19.pdf')
plt.show()