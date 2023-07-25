import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

def diff(y, t, tau, f):
    # 定义微分方程
    # dydt = -(1/tau + f(t))*y + f(t)
    dydt = -(1/tau)*y + f(t)
    return dydt

def euler_solver(y0, step_size, step_num, tau, f):
    yt = y0
    step = 0
    ans = [y0]
    while(step < step_num):
        ytt = yt + step_size * diff(yt, step*step_size, tau, f)
        yt = ytt
        ans.append(yt)
        step += 1
    return ans[0:-1]

def fused_solver(y0, step_size, step_num, tau, f):
    yt = y0
    step = 0
    ans = [y0]
    while(step < step_num):
        # ytt = (yt + step_size * f(step*step_size)) / (1 + step_size * (1/tau + f(step*step_size)))
        ytt = (yt + step_size * f(step*step_size)) / (1 + step_size * (1/tau))
        yt = ytt
        ans.append(yt)
        step += 1
    return ans[0:-1]

# 定义时间常数的取值范围和步长
#tau_values = [0.1, 1, 3.0]
TAU = 2
function = math.sin
t = np.linspace(0, 20, 200)  # 时间范围和步长
euler = euler_solver(1, 0.1, 200, TAU, function)
fused = fused_solver(1, 0.1, 200, TAU, function)

# 绘制不同时间常数下的微分方程解
y0 = 1.0  # 初始条件
y = odeint(diff, y0, t, args=(TAU, function))
plt.plot(t, y, label=f'baseline')
plt.plot(t, euler, label=f'Euler')
plt.plot(t, fused, label=f'Fused')


# 设置图形标题和标签
plt.title('Different ODE solvers')
plt.xlabel('Time')
plt.ylabel('y')

# 添加图例
plt.legend()

# 显示图形
plt.show()
