
import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sy # 代數運算，支援微分與積分
from numpy.linalg import inv # 求反矩陣

def sign(w, x):
	if np.dot(w, x) >= 0:
		return 1
	else:
		return -1

def err01(w, x, y):
	err = 0
	for i in range(len(x)):
		h = sign(w, x[i])
		if h != y[i]:
			err += 1
	return err / len(x)

# define logistic function (sigmoid)
def Logistic(x):
	return 1.0 / (1 + np.exp(-x))

# define logistic with regularization
# (ridge regression)
# Wreg = (z^T * z + λI)^-1 * z^T * y
def  LgRegularization(x, y, λ):
	# x: 200x3, x^T: 3x200
	# y: 200x1
	return np.dot(np.dot((inv(np.dot(x.transpose(), x) + λ * np.eye(dim + 1))), x.transpose()), y)

# read train data
train = np.loadtxt('hw4_train.dat')
dim = len(train[0]) - 1 # dimension
x = np.zeros((len(train), dim + 1))
y = np.zeros((len(train), 1))

for i in range(len(train)):
		y[i] = train[i][dim]
		x[i] = np.append([1], np.delete(train[i], dim))

# read test data
test = np.loadtxt('hw4_test.dat')
dim_t = len(test[0]) - 1 # dimension
xt = np.zeros((len(test), dim_t + 1))
yt = np.zeros((len(test), 1))

for i in range(len(test)):
		yt[i] = test[i][dim_t]
		xt[i] = np.append([1], np.delete(test[i], dim_t))


# No. 13
λ = 1.126
wreg = np.reshape(LgRegularization(x, y, λ), (1, dim + 1))
print("13."), print("Ein = ", err01(wreg, x, y),", Eout = ", err01(wreg, xt, yt)), print()

# No. 14 & 15
Ein_λ = []
Eout_λ = []
lamb = []
mimEin = 1
mimEout = 1

for i in range(2, -11, -1):
	λ = pow(10, i)
	wreg = np.reshape(LgRegularization(x, y, λ), (1, dim + 1))
	Ein = err01(wreg, x, y)
	Eout = err01(wreg, xt, yt)
	Ein_λ.append(Ein)
	Eout_λ.append(Eout)
	lamb.append(λ)

print("14&15."), print("Ein = ", Ein_λ), print("Eout = ", Eout_λ), print("λ = ", lamb)

fig = plt.figure()
plt.semilogx(lamb, Ein_λ,'k--', label=r"$E_{in}$", color='#8F4586')
plt.semilogx(lamb, Eout_λ, 'k', label=r"$E_{out}$", color='#64A600')
plt.legend()

############################################################################################
# No. 16&17&18 -> Validation
x_vtrain = x[:120]
y_vtrain = y[:120]
x_vtest = x[120:]
y_vtest = y[120:]
Ein_vtrain_λ = []
Ein_vtest_λ = []
Eout_λ = []
v_lamb = []
v_mimEin = 1
v_mimEout = 1


for i in range(2, -11, -1):
	λ = pow(10, i)
	wreg = np.reshape(LgRegularization(x_vtrain, y_vtrain, λ), (1, dim + 1))
	Ein_vtrain = err01(wreg, x_vtrain, y_vtrain)
	Ein_vtest = err01(wreg, x_vtest, y_vtest)
	Eout = err01(wreg, xt, yt)
	Ein_vtrain_λ.append(Ein_vtrain)
	Ein_vtest_λ.append(Ein_vtest)
	Eout_λ.append(Eout)
	v_lamb.append(λ)

print("16&17."), print("Ein_vtrain = ", Ein_vtrain_λ), print("Ein_vtest = ", Ein_vtest_λ), print("Eout = ", Eout_λ), print("λ = ", v_lamb)

fig = plt.figure()
plt.semilogx(v_lamb, Ein_vtrain_λ,'k--', label=r"$E_{train}$", color='#8F4586')
plt.semilogx(v_lamb, Ein_vtest_λ, 'k', label=r"$E_{val}$", color='#64A600')
plt.legend()

# No. 18
wreg = np.reshape(LgRegularization(x, y, 1), (1, dim + 1))
Ein = err01(wreg, x, y)
Eout = err01(wreg, xt, yt)
print("18."), print("For the wreg with optimal lambda achievd in 17. and trained in whole data:")
print("Ein = ", Ein,", Eout = ", Eout)

############################################################################################
# No. 19&20 -> CV (Cross Validation 交叉驗證)
fold = int(len(x) / 40)
A = range(2, -11, -1)
λ = pow(10 * np.ones(len(A)), np.array(A))
Ecv_λ = np.zeros(len(A))
Eout_λ = []

for i in range(len(A)):
	for j in range(fold):
		x_train = np.delete(x, np.s_[j * 40 : (j+1) * 40], axis = 0)
		y_train = np.delete(y, np.s_[j * 40 : (j+1) * 40], axis = 0)
		x_cv = x[j * 40 : (j+1) * 40]
		y_cv = y[j * 40 : (j+1) * 40]
		wreg = np.reshape(LgRegularization(x_train, y_train, λ[i]), (1, dim + 1))
		Ecv_λ[i] = Ecv_λ[i] + err01(wreg, x_cv, y_cv)
	Ecv_λ[i] = Ecv_λ[i] / 5

print("19."), print("Ecv = ", Ecv_λ), print("λ = ", λ)

fig = plt.figure()
plt.semilogx(v_lamb, Ecv_λ,'k--', label=r"$E_{cv}$", color='#8F4586')
plt.legend()

# No. 20
λ = pow(10, -8)
wreg = np.reshape(LgRegularization(x, y, λ), (1, dim + 1))
Ein = err01(wreg, x, y)
Eout = err01(wreg, xt, yt)
print("20."), print("For the wreg with optimal lambda achievd in 19. and trained in whole data:")
print("Ein = ", Ein,", Eout = ", Eout)



plt.show()

