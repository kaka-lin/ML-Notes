import numpy as np
import matplotlib.pyplot as plt


'''
Learning Model: Positive and Negative Rays 
=> hypotheses : h(x) = s*sing(x-theta)
The model is frequently named the "decision stump" model
the VC dimension of the decision stump model is 2 
(break point is 3)
'''

size = 10
lamda = 0.8
repeat = 5000

# def sign(x)
def sign(x):
	if x >= 0:
		return 1
	else:
		return -1
# def h(x) = s*sing(x-theta)
def h(s, theta, x):
	return s * sign(x - theta)

# def Eout = 0.5 + 0.3s * (abs(theta) - 1)
def eOut(s, theta):
	return 0.5 + 0.3 * s * (abs(theta) - 1)

Ein = np.zeros((1, repeat))[0]
Eout = np.zeros((1, repeat))[0]

for times in range(repeat):
	# Generate x by a uniform distribution in [−1, 1] and size is 10
	x = np.sort(np.random.uniform(-1, 1, size))

	'''
	Generate y by s ̃(x) + noise where s ̃(x) = sign(x) 
	and the noise flips the result with 20% probability.
	'''

	# Generate s ̃(x) by sign(x) 
	s_tilde = []
	for i in x:
		s_tilde.append(sign(i))

	# Generate random value in [0, 1] and if value < 0.2, add noise	
	y = s_tilde * np.where(np.random.random(size) < (1-lamda), -1, 1)

	'''
	The chosen dichotomy stands for a combination of some ‘spot’ (range of θ) and s,
	 and commonly the median of the range is chosen as the θ that realizes the dichotomy.
	'''

	# Generate theta
	# 首先將data兩端加上-1, 1
	x1 = np.append(np.append([-1], x),[1])
	# 計算theta : 兩點之中值
	theta = [(x1[i] + x1[i+1]) / 2 for i in range(size + 1)] # List Comprehensions

	min_err_rate = 1
	best_theta = 1
	best_s = 0
	for i in range(size + 1):
		err = 0 # s = 1
		err_rate = 0
		err_nag = 0 # s = -1
		err_nag_rate = 0
		for j in range(size):
			g = h(1, theta[i], x[j])
			g_nag = h(-1, theta[i], x[j])
			if g != y[j]:
				err += 1
			if g_nag != y[j]:
				err_nag += 1
		err_rate = err / size
		err_nag_rate = err_nag / size

		if err_rate < err_nag_rate:
			if err_rate < min_err_rate:
				min_err_rate = err_rate
				best_theta = theta[i]
				best_s = 1	
		else:
			if err_nag_rate < min_err_rate:
				min_err_rate = err_nag_rate
				best_theta = theta[i]
				best_s = -1

	Ein[times] = min_err_rate * 100 # percent
	Eout[times] = eOut(best_s, best_theta) * 100 # percent

aver_Ein = round(sum(Ein) / repeat, 2) # round: 四捨五入, 2:取到小數以下第二位
aver_Eout = round(sum(Eout) / repeat, 2)

print("average Ein: ", aver_Ein, "%")
print("average Eout: ", aver_Eout, "%")

bins_1 = range(60)
plt.figure(1)
plt.hist(Ein, bins_1, histtype = 'bar', rwidth = 0.8)
plt.xlabel("Error rate (%)")
plt.ylabel("Frequency")
plt.title("Average Ein = " + str(aver_Ein) + "%", fontsize = 10)
plt.savefig('hw2_17.png')
#plt.savefig('hw2_17.pdf')

bins_2 = range(100)
plt.figure(2)
plt.hist(Eout, bins_2, histtype = 'bar', rwidth = 0.8)
plt.xlabel("Error rate (%)")
plt.ylabel("Frequency")
plt.title("Average Eout = " + str(aver_Eout) + "%", fontsize = 10)
plt.savefig('hw2_18.png')
#plt.savefig('hw2_18.pdf')

plt.show()
