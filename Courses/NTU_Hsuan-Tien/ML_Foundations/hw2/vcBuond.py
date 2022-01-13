import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from sympy.solvers import solve
from sympy import Symbol
# SymPy 計算各種方程式，跟matlab差不多
# SymPy支持符號計算、高精度計算、模式匹配、繪圖、解方程、微積分、組合數學、離散數學、幾何學、機率與統計、物理學等方面的功能

dvc = 50
target = 20000
delta = 0.05
time = np.arange(target*0.5, target*1.5, target/100)

def origin_vc_bound(N):
	return np.sqrt((8.0 / N) * np.log(4 * pow(2 * N, dvc) / delta))
	
def variant_vc_bound(N):
	return np.sqrt((16.0 / N) * np.log((2 * pow(N, dvc)) / np.sqrt(delta))) 	

def rademacher_penalty_bound(N):
	return np.sqrt((2 * (np.log(2 * N) + dvc * np.log(N))) / N ) + np.sqrt((2.0 / N) * np.log(1 / delta)) + (1.0 / N)
	
def parrondo_vandenBroek(N):
	x = Symbol('x')
	e = solve(sy.sqrt((1.0 / N) * (2 * x + np.log(6) + dvc * np.log(2 * N) - np.log(delta))) - x, x)
	return e
    #return e - np.sqrt((1.0 / N) * (2 * e + np.log(6) + dvc * np.log(2 * N) - np.log(delta)))
    
def devroye(N):
	y = Symbol('y')
	e = solve(sy.sqrt((1.0 / (2 * N)) * (4 * y * (1 + y) + np.log(4) + 2  * dvc * np.log(N) - np.log(delta))) - y, y)
	return e 
    #return e - np.sqrt((1.0 / (2 * N)) * (4 * e * (1 + e) + np.log(4) + 2  * dvc * np.log(N) - np.log(delta)))


print(origin_vc_bound(target))
print(variant_vc_bound(target))
print(rademacher_penalty_bound(target))
print(parrondo_vandenBroek(target))
print(devroye(target))