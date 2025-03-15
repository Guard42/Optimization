<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# 最优化理论与方法学习记录（Work In Progress）
⚠️⚠️⚠️本仓库目前还在施工中，README内的所有内容处于画大饼状态，请见谅⚠️⚠️⚠️

在学习最优化理论与方法的过程中，发现了[zhangchenhaoseu/Optimization](https://github.com/zhangchenhaoseu/Optimization)这个repo。由于原仓库作者应该不会继续更新该仓库，出于学习目的（和原始仓库README欠排版的问题），fork了这个repo。
本仓库会将原仓库代码保留在exer文件夹中，在src文件夹中将会实现自变量维度为`n`的算法复现，并将实现的算法与求解器结果进行集成的对比，其他在学习中遇到的算法也会尽量在本仓库实现。
除此之外本仓库还会完成北京大学[文再文](http://faculty.bicmr.pku.edu.cn/~wenzw/index.html)教授开设的[《最优化方法》](http://faculty.bicmr.pku.edu.cn/~wenzw/opt-2024-fall.html)的程序大作业题的算法实现和报告，其题目如下：

> Consider the $\ell_1$-regularized problem
>
> ```math
> 
>\quad \min_{x} \quad \frac{1}{2}\|A x - b\|_{2}^{2} + \mu\|x\|_{1},
> 
> ```
>
> where $A \in \mathbb{R}^{m \times n}$, $b \in \mathbb{R}^{m}$ and $\mu > 0$ are given.

# Contact
Feel free to reach out

Author: Junwei Li

Mail: mailto:junweilee@njust.edu.cn

# Updates
- [March 15, 2025]  README.md updated for repo roadmap.

# Files
**`/src`**:包含各类算法复现和收敛次数对比可视化

- 1.`坐标轴交替下降法`，求解二次函数，分别使用坐标轴方向作为下降方向，精确线搜索计算步长；
- 2.`改进的坐标轴交替下降法`，求解二次函数，使用坐标轴下降的合成方向作为最终下降方向，精确线搜索计算步长；
- 3.`最速下降法`，求解二次函数，使用负梯度方向作为下降方向，精确线搜索计算步长；
- 4.`牛顿法`，求解凸二次函数，使用泰勒展式拟合后的函数梯度作为下降方向，步长为纯牛顿步长；
- 5.`牛顿法`，求解四次函数，使用泰勒展式拟合后的函数梯度作为下降方向，步长为纯牛顿步长；
- 6.`修正牛顿法`，求解四次函数，使用两种方法（1.对hesse阵增加单位阵；2.特征值分解，对不满足要求的特征值进行调整）实现矩阵正定得到下降方向，步长为纯牛顿步长；
- 7.`拟牛顿DFP方法`，求解四次函数，对hesse阵增加单位阵作为初始矩阵，DFP公式计算Hk+1进而得下降方向，步长为纯牛顿步长；
- 8.`拟牛顿BFGS方法`，求解四次函数，对hesse阵增加单位阵作为初始矩阵，BFGS公式计算Bk+1进而得下降方向，步长为纯牛顿步长；
- 9.`拟牛顿SR-1方法`，求解四次函数，对hesse阵增加单位阵作为初始矩阵，SR-1公式计算Bk+1进而得下降方向，步长为纯牛顿步长；
- 10.`线性共轭梯度法`，求解凸二次函数，根据梯度信息+目标函数的二次型矩阵，计算得到下降方向和步长；
- 11.`非线性FR共轭梯度法`，求解四次函数，根据梯度信息计算下降方向，使用Goldstein方法得到步长；
- 12.`非线性PRP共轭梯度法`，求解四次函数，根据梯度信息计算下降方向，使用Goldstein方法得到步长；

**`/src_homework`**：包含程序大作业题的算法实现和数值实验。

- 1. `原始问题的次梯度法`:Subgradient method for the primal problem.
 Read the subgradient method in
 [Subgradient Method](http://bicmr.pku.edu.cn/˜wenzw/opt2015/lect-sgm.pdf)
- 2. `平滑原始问题的梯度法`:Gradient method for the smoothed primal problem.
 Read the smoothing technique in
 [Lecture: Smoothing](http://bicmr.pku.edu.cn/˜wenzw/opt2015/Smoothing.pdf)
- 3. `平滑原始问题的快速（Nesterov/加速）梯度法`:Fast (Nesterov/accelerated) gradient method for the smoothed primal problem.
 Read the acceleration techniques in
 [Lecture: Fast Proximal Gradient Methods](http://bicmr.pku.edu.cn/˜wenzw/opt2015/slides-fgrad.pdf)
- 4. `原始问题的近端梯度法`:Proximal gradient method for the primal problem.
 Read [Proximal gradient method](http://bicmr.pku.edu.cn/˜wenzw/opt2015/lect-proxg.pdf)
- 5. `原始问题的快速近端梯度法`:Fast proximal gradient method for the primal problem.
 Read the acceleration techniques in
 [Lecture: Fast Proximal Gradient Methods](http://bicmr.pku.edu.cn/˜wenzw/opt2015/slides-fgrad.pdf)
- 6. `对偶问题的增广拉格朗日法`:Augmented Lagrangian method for the dual problem.
 Read the augmented Lagrangian method in
 [Lecture: Proximal Point Method](http://bicmr.pku.edu.cn/˜wenzw/opt2015/lect-prox-point.pdf)
- 7. `对偶问题的乘数交替方向法`:Alternating direction method of multipliers for the dual problem.
 Read the augmented Lagrangian method in
 [Lecture: Proximal Point Method](http://bicmr.pku.edu.cn/˜wenzw/opt2015/lect-prox-point.pdf)
- 8. `原始问题的乘数交替方向法及其线性化`:Alternating direction method of multipliers with linearization for the primal problem.
 Read the ADMM in [Douglas-Rachford method, ADMM and PDHG](http://bicmr.pku.edu.cn/˜wenzw/opt2015/lect-admm.pdf).
 Read the ADMM with a single gradient (or proximal graident) step in pages 15 and 16 in
 [Lecture on Alternating direction method of multipliers(ADMM)](http://bicmr.pku.edu.cn/˜wenzw/opt2015/lect-admm-part2.pdf)
- 9. `对偶问题的近端点法`:Proximal point method for the dual problem
- 10. `原始问题的块坐标法`:Block coordinate method for the primal problem
 
# References
- [Original Repo](https://github.com/zhangchenhaoseu/Optimization)
- [Bilibili - 崔雪婷老师的视频](https://space.bilibili.com/507629580/video.)
- [北京大学课程 - 最优化方法 (2024年秋季)](http://faculty.bicmr.pku.edu.cn/~wenzw/opt-2024-fall.html)
- [最优化方法复习笔记（一）梯度下降法、精确线搜索与非精确线搜索（推导+程序）](https://zhuanlan.zhihu.com/p/271088190)
- [最优化方法复习笔记（三）牛顿法及其收敛性分析](https://zhuanlan.zhihu.com/p/293951317)
- [最优化方法复习笔记（四）拟牛顿法与SR1,DFP,BFGS三种拟牛顿算法的推导与代码实现](https://zhuanlan.zhihu.com/p/306635632)
- [最优化方法复习笔记（六）共轭梯度法](https://zhuanlan.zhihu.com/p/338838078)
