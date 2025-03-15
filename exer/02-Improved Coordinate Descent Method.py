'''
1.坐标轴交替下降法，求解二次函数，分别使用坐标轴方向作为下降方向，精确线搜索计算步长；

Coordinate Descent Method: Solve quadratic functions using coordinate axes
as descent directions, with exact line search for step calculation.
'''

import numpy as np
import sympy as sp
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

class CoordinateDescentMethod:
    '''
    坐标轴交替下降法，求解二次函数，分别使用坐标轴方向作为下降方向，精确线搜索计算步长；
    '''

    def __init__(self, dimension=2, max_iter=100, epsilon=1e-4, f=None, seed=42):
        self.dimension = dimension
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.f = f
        self.seed = seed
        self.x = []
        self.xs = sp.symbols('x1:%d' % (dimension + 1))
        self.xGrad = []
        self.xs_ = sp.Matrix(sp.symbols('x1:%d' % (dimension + 1)))

        for i in range(self.dimension):
            self.x.append(sp.Symbol('x%d' % (i + 1)))
            self.xGrad.append(sp.diff(f, self.x[i]))

        # 使用 sympy.lambdify 将符号表达式转换为可执行函数
        self.func = sp.lambdify(self.xs, self.f, modules="numpy")

    def getFunc(self, x=np.zeros((2, 1))):
        x_vals = {str(self.x[j]): x[j, 0] for j in range(self.dimension)}  # 使用符号变量作为键
        f_value = self.f.evalf(subs=x_vals)
        return float(f_value)  # 确保返回值是浮点数

    def getFunc_(self, x=np.zeros((2, 1))):
        # 使用生成的可执行函数计算函数值
        return self.func(*x.flatten())

    def getGrad(self, x=np.zeros((2, 1))):  # 求梯度
        gradients = []
        x_vals = {str(self.x[j]): x[j, 0] for j in range(self.dimension)}  # 使用符号变量作为键
        for i in range(self.dimension):
            g_x_i = self.xGrad[i].evalf(subs=x_vals)  # 计算梯度
            gradients.append(float(g_x_i))  # 将 sympy.Float 转换为 float

        grad_2norm = np.linalg.norm(np.array(gradients))  # 计算梯度的范数
        return grad_2norm

    def getStepLength(self, x=np.zeros((2, 1)), search_dir=1):
        # 求解该方向上的极小值点
        gxS = self.xGrad[search_dir - 1]  # 获取当前方向的梯度表达式
        xS = self.x[search_dir - 1]  # 获取当前方向的符号变量

        # 解析梯度为零的点
        xSfunction = sp.solve(gxS, xS)[0]

        # 准备 subs 字典，将其他变量的值代入
        x_vals = {str(self.x[j]): x[j, 0] for j in range(self.dimension)}  # 使用符号变量作为键，数值作为值

        # 计算步长
        step = float(xSfunction.evalf(subs=x_vals)) - x[search_dir - 1, 0]  # 确保结果为浮点数
        return step

    def cdm(self, start_x=np.zeros((2, 1))):
        k = 0
        x_ = start_x
        x_list = [[] for _ in range(self.dimension)]
        for i in range(self.dimension):
            x_list[i].append(float(x_[i, 0]))  # 初始化 x_list
        f_value_list = [self.getFunc(x_)]  # 初始化 f_value_list

        while self.getGrad(x_) > self.epsilon and k <= self.max_iter:
            search_dir = k % 3 + 1
            if search_dir == 3:
                step1 = self.getStepLength(x_, 1)
                step2 = self.getStepLength(x_, 2)
                x_[0, 0] += step1
                x_[1, 0] += step2
                x_list[0].append(float(x_[0, 0]))
                x_list[1].append(float(x_[1, 0]))
            else:
                step = self.getStepLength(x_, search_dir)
                x_[search_dir - 1, 0] += step  # 更新 x_
                x_list[search_dir - 1].append(float(x_[search_dir - 1, 0]))  # 更新 x_list
            f_value_list.append(self.getFunc(x_))  # 更新 f_value_list
            k += 1

        f_value = self.getFunc(x_)
        print("The optimal solution is:", x_)
        print("The optimal function value is:", f_value)
        print("The number of iterations is:", k)
        print("The x iteration list:", x_list)
        print("The f_value iteration list:", f_value_list)

        return x_, f_value, k, x_list, f_value_list

    def visualize_2d_optimization(self, x_list, f_value_list):
        """
        2D可视化优化过程和结果
        """
        # 提取优化路径的 x1 和 x2 坐标
        x1_path = x_list[0]  # x1 的路径
        x2_path = x_list[1]  # x2 的路径

        # 生成网格数据
        x1_vals = np.linspace(-100, 100, 400)
        x2_vals = np.linspace(-100, 100, 400)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        F = self.func(X1, X2)  # 使用 self.func 快速计算网格点上的函数值

        # 绘制等高线图
        plt.figure(figsize=(8, 6))
        contour = plt.contour(X1, X2, F, levels=50, cmap='viridis')
        plt.colorbar(contour, orientation='horizontal', shrink=0.8)

        # 使用学术期刊推荐的色盘
        colors = plt.cm.tab20(np.linspace(0, 20, len(f_value_list)))  # 使用 tab10 色盘

        x1_index = 0
        x2_index = 0
        # 绘制优化路径，并添加箭头
        for i in range(len(f_value_list) -1):
            # 当前点和下一个点
            if x1_index == x2_index:
                current_point = (x1_path[x1_index], x2_path[x2_index])
                x1_index += 1
                next_point = (x1_path[x1_index], x2_path[x2_index])
            elif x1_index > x2_index:
                current_point = (x1_path[x1_index], x2_path[x2_index])
                x2_index += 1
                next_point = (x1_path[x1_index], x2_path[x2_index])
            print(f'line {i}:({current_point[0]},{current_point[1]})->({next_point[0]},{next_point[1]})')
            # 绘制路径线段
            plt.plot([current_point[0], next_point[0]], [current_point[1], next_point[1]],
                     marker='o', color=colors[i], linestyle='-', linewidth=1.5)

            # 添加箭头
            plt.annotate("", xy=next_point, xytext=current_point,
                         arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5))

        plt.title('2D Optimization Path', fontsize=14)
        plt.xlabel('x1', fontsize=12)
        plt.ylabel('x2', fontsize=12)
        plt.legend([f"Iteration {i}" for i in range(len(x1_path) - 1)], fontsize=10, loc='best')
        plt.grid(True)
        plt.savefig('../fig/2.改进坐标轴交替下降法2D.png', dpi=300)  # 保存图像，设置高分辨率
        plt.show()
        plt.close()

    def visualize_3d_optimization(self, x_list, f_value_list):
        """
        3D可视化优化过程和结果
        """
        # 提取优化路径的 x1 和 x2 坐标
        x1_path = x_list[0]  # x1 的路径
        x2_path = x_list[1]  # x2 的路径

        # 生成网格数据
        x1_vals = np.linspace(-100, 100, 100)
        x2_vals = np.linspace(-100, 100, 100)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        F = self.func(X1, X2)  # 使用 self.func 快速计算网格点上的函数值

        # 绘制 3D 曲面图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.8)

        # 使用学术期刊推荐的色盘
        colors = plt.cm.tab20(np.linspace(0, 1, len(f_value_list)))  # 使用 tab20 色盘

        x1_index = 0
        x2_index = 0
        # 绘制优化路径，并添加箭头
        for i in range(len(f_value_list) - 1):
            # 当前点和下一个点
            if x1_index == x2_index:
                current_point = (x1_path[x1_index], x2_path[x2_index], f_value_list[i])
                x1_index += 1
                next_point = (x1_path[x1_index], x2_path[x2_index], f_value_list[i + 1])
            elif x1_index > x2_index:
                current_point = (x1_path[x1_index], x2_path[x2_index], f_value_list[i])
                x2_index += 1
                next_point = (x1_path[x1_index], x2_path[x2_index], f_value_list[i + 1])

            print(
                f'line {i}:({current_point[0]},{current_point[1]},{current_point[2]}) -> ({next_point[0]},{next_point[1]},{next_point[2]})')

            # 绘制路径线段
            ax.plot([current_point[0], next_point[0]],
                    [current_point[1], next_point[1]],
                    [current_point[2], next_point[2]],
                    marker='o', color=colors[i], linestyle='-', linewidth=1.5)

            # 添加箭头
            ax.quiver(current_point[0], current_point[1], current_point[2],
                      next_point[0] - current_point[0],
                      next_point[1] - current_point[1],
                      next_point[2] - current_point[2],
                      color=colors[i], arrow_length_ratio=0.1)

        ax.set_title('3D Optimization Path', fontsize=14)
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_zlabel('Function Value', fontsize=12)
        plt.colorbar(surface, shrink=0.8)
        plt.legend([f"Iteration {i}" for i in range(len(f_value_list) - 1)], fontsize=10, loc='best')
        plt.grid(True)
        plt.savefig('../fig/2.改进坐标轴交替下降法3D.png', dpi=300)  # 保存图像，设置高分辨率
        #plt.show()
        plt.close()

if __name__ == "__main__":
    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    f = 2 * pow(x1, 2) - 2 * pow(x2, 2) + 2 * x1 * x2 - 5
    gx1 = sp.diff(f, x1)  # x1偏导
    gx2 = sp.diff(f, x2)  # x2偏导

    # 初始化优化算法类
    cdm = CoordinateDescentMethod(f=f)

    # 调用优化算法
    start_x = np.array([[15], [30]])  # 初始点
    x_opt, f_opt, iterations, x_list, f_value_list = cdm.cdm(start_x=start_x)

    print(f"Optimal point: {x_opt}")
    print(f"Optimal function value: {f_opt}")
    print(f"Iterations: {iterations}")

    # 可视化优化过程和结果
    cdm.visualize_2d_optimization(x_list, f_value_list)
    cdm.visualize_3d_optimization(x_list, f_value_list)