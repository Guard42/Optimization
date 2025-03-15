'''
6.修正牛顿法，求解四次函数，使用两种方法
----A.对hesse阵增加单位阵；
----B.特征值分解，对不满足要求的特征值进行调整
实现矩阵正定得到下降方向，步长为纯牛顿步长；

Modified Newton's Method: Solve quartic functions using two methods
(1. Adding the identity matrix to the Hessian;
2. Eigenvalue decomposition, adjusting non-satisfying eigenvalues)
to achieve matrix positive definiteness for the descent direction, with pure Newton step length.
'''