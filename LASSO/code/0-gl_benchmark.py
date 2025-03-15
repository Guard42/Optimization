import numpy as np
import cvxpy as cvx
import gurobipy as grb
import time

# Define the problem
n = 1024
m = 512

# Generate data
A = np.random.randn(m, n)
u = np.random.rand(n, 1)
b = A @ u

mu = 0.001

x0 = np.random.rand(n)

def errfun(x1, x2):
    return np.linalg.norm(x1 - x2) / (1 + np.linalg.norm(x1))

# CVX calling MOSEK
opts1 = {}
start_time = time.time()
x1, out1 = cvx.Problem(A, b, mu, x0, options=opts1).solve()
t1 = time.time()

# CVX calling Gurobi
opts2 = {}
start_time = time.time()
x2, out2 = cvx.Problem(A, b, mu, x0, method='GUROBI', options=opts2).solve()
t2 = time.time()

# Call MOSEK directly
opts3 = {}
start_time = time.time()
x3, out3 = grb.minimize('2*||A*x-b||^2 + mu*||x||1', {'x': x0}, options=opts3)
t3 = time.time()

# Call Gurobi directly
opts4 = {}
start_time = time.time()
x4, out4 = grb.minimize('2*||A*x-b||^2 + mu*||x||1', {'x': x0}, options=opts4)
t4 = time.time()

# Other approaches
opts5 = {}
start_time = time.time()
x5, out5 = grb.minimize('2*||A*x-b||^2 + mu*||x||1', {'x': x0}, options=opts5)
t5 = time.time()

# Print comparison results with CVX-call-mosek
print(f'CVX-call-gurobi: cpu: {t2}:.2f, err-to-cvx-mosek: {errfun(x1, x2):.2e}\n'
      f'     call-mosek: cpu: {t3}:.2f, err-to-cvx-mosek: {errfun(x1, x3):.2e}\n'
      f'    call-gurobi: cpu: {t4}:.2f, err-to-cvx-mosek: {errfun(x1, x4):.2e}\n'
      f'     methodxxxx: cpu: {t5}:.2f, err-to-cvx-mosek: {errfun(x1, x5):.2e}\n')