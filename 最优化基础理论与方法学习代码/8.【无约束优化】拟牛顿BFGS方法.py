# 靡不有初，鲜克有终
# 开发时间：2022/10/25 13:13

"""
演示BFGS拟牛顿法解决一个凸函数的最优化问题的例子
优化函数：三次函数，n=2
        f(x_1,x_2)=(x_1)**4-2(x_1)**3+(x_2)**2-(x_1)*(x_2)+1 （非凸、有最小点）
"""

import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import eig

x1 = sp.Symbol('x1')
x2 = sp.Symbol('x2')
f = pow(x1, 4) - 2 * pow(x1, 3) + pow(x2, 2) - x1 * x2 + 1
gx1 = sp.diff(f, x1)
gx2 = sp.diff(f, x2)
gx1x1 = sp.diff(gx1, x1)
gx1x2 = sp.diff(gx1, x2)
gx2x1 = sp.diff(gx2, x1)
gx2x2 = sp.diff(gx2, x2)
mtx_H = np.matrix([[gx1x1, gx1x2], [gx2x1, gx2x2]])  # Hesse阵
lambda_ = 40  # 增加单位阵的乘数


def function_value(vector):  # 求函数值
    f_value = f.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    return f_value


def gradient(vector):  # 求梯度向量及模长
    g_x1 = gx1.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    g_x2 = gx2.evalf(subs={x1: vector[0][0], x2: vector[1][0]})
    vector_g = np.array([[g_x1], [g_x2]])
    modulus = (pow(g_x1, 2) + pow(g_x2, 2)) ** 0.5
    return [vector_g, modulus]


def hesse(vector):  # 求海塞阵修正后的数值,矩阵形式给出
    gx1x1_value = float(gx1x1.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    gx1x2_value = float(gx1x2.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    gx2x1_value = float(gx2x1.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    gx2x2_value = float(gx2x2.evalf(subs={x1: vector[0][0], x2: vector[1][0]}))
    mtx_H_value = np.matrix([[gx1x1_value, gx1x2_value], [gx2x1_value, gx2x2_value]])
    mtx_H_value = mtx_H_value + lambda_*np.eye(2)
    return mtx_H_value


def generate_grid(min_x_1, max_x_1, min_x_2, max_x_2, delta):  # 三维网格底图，参数含范围和颗粒度
    x_1_array = np.arange(min_x_1, max_x_1, delta)
    x_2_array = np.arange(min_x_2, max_x_2, delta)
    x1_array, x2_array = np.meshgrid(x_1_array, x_2_array)
    f_value_array = pow(x1_array, 4) - 2 * pow(x1_array, 3) + pow(x2_array, 2) - x1_array * x2_array + 1
    return x1_array, x2_array, f_value_array


def BFGS_quasi_newton_method(vector_initial, epsilon):  # BFGS拟牛顿法
    k = 0
    n = 0  # n步重启
    vector_x_pre = vector_initial
    B_initial = hesse(vector_x_pre)  # 矩阵B_0的计算（或者修正）要体现一些二阶信息，计算成本也要低，当然信息越多越好
    B_pre = B_initial  # B_0
    B_pre = B_pre.astype(np.float)
    B_inverse_pre = np.linalg.inv(B_pre)  # 初始的B逆

    x_1_lst = [vector_x_pre[0][0]]
    x_2_lst = [vector_x_pre[1][0]]
    f_value_lst = [function_value(vector_x_pre)]

    while float(gradient(vector_x_pre)[1]) > epsilon and k <= 300:
        '''if n == 10:  # n 步重启
            B_pre = hesse(vector_x_pre)
            B_pre = B_pre.astype(np.float)
            B_inverse_pre = np.linalg.inv(B_pre)
            n = 0'''
        vector_x_new = np.array(vector_x_pre - np.dot(B_inverse_pre, gradient(vector_x_pre)[0]))
        x_1_lst.append(vector_x_new[0][0])
        x_2_lst.append(vector_x_new[1][0])
        f_value_lst.append(function_value(vector_x_new))

        g_pre = gradient(vector_x_pre)[0]
        g_new = gradient(vector_x_new)[0]
        y = g_new - g_pre
        s = vector_x_new - vector_x_pre

        B_inverse_new = np.dot((np.eye(2)-np.dot(s,y.T)/np.dot(s.T,y).item()),np.dot(B_inverse_pre,(np.eye(2)-np.dot(s,y.T)/np.dot(s.T,y).item()).T))+np.dot(s,s.T)/np.dot(s.T,y).item() # B的逆，使用Sherman-Morrison公式推导出来的,如果1步重启的话这个式子压根就不会起到效果
        B_inverse_pre = B_inverse_new

        vector_x_pre = vector_x_new
        k = k + 1
        n = n + 1

    f_value = function_value(vector_x_pre)
    return vector_x_pre, f_value, k, x_1_lst, x_2_lst, f_value_lst


def plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst):  # 绘制二维图像并保存
    plt.figure()
    plt.contourf(X1, X2, F, 10)
    plt.colorbar(orientation='horizontal', shrink=0.8)
    plt.plot(x_1_lst, x_2_lst, c='r', linewidth=1.5)
    plt.savefig('8.BFGS拟牛顿法2D求解四次函数.png')
    plt.show()


def plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst,f_value_lst):  # 绘制三维图像并保存
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.plot_surface(X1, X2, F, rstride=4, cstride=4, cmap='jet', alpha=0.8)
    ax.plot3D(x_1_lst, x_2_lst, f_value_lst, c='r', linewidth=1.5)
    plt.colorbar(p, shrink=0.8)
    plt.savefig('8.BFGS拟牛顿法3D求解四次函数.png')
    plt.show()


if __name__ == "__main__":
    vector = np.array([[-0.5], [1.5]])
    vector_x_opt, value, times, x_1_lst, x_2_lst, f_value_lst = BFGS_quasi_newton_method(vector, 0.001)
    X1, X2, F = generate_grid(-1.3, 2, -2, 2, 0.01)  # 生成用于三维展示的函数面
    print("最优点坐标", vector_x_opt[0][0], ',', vector_x_opt[1][0])
    print("最优值", value)
    print('迭代次数', times)
    plot_2D_figure(X1, X2, F, x_1_lst, x_2_lst)
    plot_3D_figure(X1, X2, F, x_1_lst, x_2_lst, f_value_lst)
