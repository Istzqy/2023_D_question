# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 15:27:11 2024

@author: admin
"""

#非线性最小二乘拟合
#基于Pex8_10_1进行修改,将lambda定义函数方法改为通过def 定义函数
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# # 定义待拟合函数(Logitic人口增长预测模型)
def funcPop(t, r, xm):
    return xm/(1+(xm/7869.34-1)*np.exp(-r*(t-2010)))

a=[]; b=[];
with open("Popdata_Shumo.txt") as f:    #打开文件并绑定对象f
    s=f.read().splitlines()  #返回每一行的数据
for i in range(0, len(s),2):  #读入奇数行数据
    d1=s[i].split("\t")
for j in range(len(d1)):
        if d1[j]!="": a.append(eval(d1[j]))  #把非空的字符串转换为年代数据
for i in range(1, len(s), 2):  #读入偶数行数据
  d2=s[i].split("\t")
for j in range(len(d2)):
        if d2[j] != "": b.append(eval(d2[j])) #把非空的字符串转换为人口数据
c=np.vstack((a,b))  #构造两行的数组
np.savetxt("Popdata_Shumo_2.txt", c)  #把数据保存起来供下面使用

#统计2010至2020该区域的人口年均增长率
#0.0195、0.0122、0.0089、0.0108、0.0041、0.00798、0.005014633、0.002693655、0.002693655、0.002711282、0.000964685
#均值：0.007489


#bd=((0, 200), (0.01,12000))  #约束两个参数的下界和上界
popt, pcov=curve_fit(funcPop, a, b)
print(popt); print("2060年的预测值为：", funcPop(2060, *popt))

#绘图
fig,ax=plt.subplots()
ax.plot(c[0],c[1],'b-')
ax.legend(r'$original\ values$')
poly_y = [funcPop(xx,popt[0],popt[1]) for xx in c[0]]
ax.plot(c[0],poly_y,'r--')
ax.legend(r'$polyfit\ values$')
plt.show()


t2 = np.linspace(2010, 2060,51)  # 时间间隔从2021到2060，将其均匀分成40个点
poly_y2 = [funcPop(xx,popt[0],popt[1]) for xx in t2]
plt.plot(t2, poly_y2)  
plt.xlabel('Time')
plt.ylabel('Population')
#————————————————
# 版权声明：本文为CSDN博主「hover_load」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/qq_45268474/article/details/108065667



##################################################################################
##################################################################################
#线性最小二乘拟合
import numpy as np
d=np.loadtxt("Popdata_Shumo_2.txt")    #加载文件中的数据
t0=d[0];x0=d[1]  #提取年代数据及对应的人口数据

b=np.diff(x0)/10/x0[:-1]  #构造线性方程组的常数项列
a=np.vstack([np.ones(len(x0)-1),-x0[:-1]]).T #构造线性方程组系数矩阵
rs=np.linalg.pinv(a)@b;  r=rs[0]; xm=r/rs[1]
print("人口增长率r和人口最大值xm的拟合值分别为", np.round([r,xm],4))
xhat=xm/(1+(xm/3.9-1)*np.exp(-r*(2060-2010)))  #求预测值
print("2010年的预测值为：",round(xhat,4))







