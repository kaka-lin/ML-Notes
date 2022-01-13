import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sy # 代數運算，支援微分與積分

def E_uv(u, v):
	return math.exp(u) + math.exp(2 * v) + math.exp(u * v) +\
	       pow(u, 2) - (2 * u * v) + (2 * pow(v, 2)) - (3 * u) - (2 * v)

u = sy.Symbol('u')
v = sy.Symbol('v')
E = sy.exp(u) + sy.exp(2 * v) + sy.exp(u * v) + pow(u, 2) - (2 * u * v) + (2 * pow(v, 2)) - (3 * u) - (2 * v)
Eu = sy.diff(E, u)
Ev = sy.diff(E, v)
print("E(u,v) = ", E)
print("Eu = ", Eu)
print("Ev = ", Ev)

# Gradient Descent Algorithm
# (ut+1, vt+1) = (ut, vt) − η∇E(ut, vt)
# (u0,v0) = (0,0)
# η = 0.01
# 補充： 計算Eu(u = 0, v = 0)
#       -> Eu.subs(dict({u:0,v:0}))

def GD(ut, vt):
	(ut, vt) = (ut - 0.01 * Eu.subs(dict({u:ut, v:vt})), vt - 0.01 * Ev.subs(dict({u:ut, v:vt})))
	return (ut, vt)

(a, b) = GD(0, 0)
for i in range(4):
	(a, b) = GD(a, b)

print("(u5,v5) =", (a, b))
print("E(u5,v5) =", round(E_uv(a, b), 3)) # round() -> 取小數第幾位




