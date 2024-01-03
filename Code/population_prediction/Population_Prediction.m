
%参考链接：https://blog.csdn.net/weixin_46308081/article/details/115640828
%% Logistic模型预测
% 利用线性最小二乘进行参数拟合
%存在问题：预测的Xm较小，预测时间长后模型预测不准，时间长后基本维持不变
clc
clear
close all
x=[7869.34	8022.99	8119.81	8192.44	8281.09	8315.11	8381.47	8423.50	8446.19	8469.09	8477.26];%人口
n=length(x);
t=0:1:n-1;
rk=zeros(1,n);
rk(1)=(-3*x(1)+4*x(2)-x(3))/2;
rk(n)=(x(n-2)-4*x(n-1)+3*x(n))/2;
for i=2:n-1
    rk(i)=(x(i+1)-x(i-1))/2;
end
rk=rk./x;
p=polyfit(x,rk,1);
b=p(2);
a=p(1);
r0=b;
xm=-r0/a;
%输出
pnum=zeros(n,1);
for i=0:1:n-1
    pnum(i+1)=xm/(1+(xm/x(1)-1)*exp(-r0*i));
end
year1=2010:2020;
plot(year1,pnum,'r--o',year1,x,'k-*')
xlabel('年份')
ylabel('人口数(万人)')
legend('实际人口','拟合人口')

figure(2)
fnum=zeros(n+39,1);
for i=0:1:n+39
    fnum(i+1)=xm/(1+(xm/x(1)-1)*exp(-r0*i));
end
year2=2010:2060;
plot(year2,fnum,'r--o')
title('Logistic模型拟合图');
xlabel('年份');
ylabel('人口数(万人)');
legend('线性最小二乘拟合预测人口数量');

%% 非线性最小二乘预测
%代码如下
clc
clear
close all
t = [2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020];
p = [7869.34 8022.99 8119.81 8192.44 8281.09 8315.11 8381.47 8423.50 8446.19 8469.09 8477.26];
t = t-2010; %整体减去1780
para = [10000,0.02]; %待定参数x的初值（自己根据实际情况给出初值，之后再不断调整；其中第一个参数为最大人口数，第二个参数为人口增长率）
fitPara = lsqcurvefit('Logitic_Pop',para,t,p); %使用函数求得最终的（xm，r）
p1 = Logitic_Pop(fitPara,t);
figure(1)
plot(t+2010,p,'o',t+2010,p1,'-r*')
title('Logistic模型拟合图')
xlabel('年份');
ylabel('人口数(万人)');
legend('实际人口','拟合人口')

figure(2)
p2=zeros(51,1);
for i=0:1:50
    p2(i+1)=fitPara(1)/(1+(fitPara(1)/p(1)-1)*exp(-fitPara(2)*i));
end
t2=2010:2060;
t2=t2-2010;
p2=Logitic_Pop(fitPara,t2);
plot(t2+2010,p2,'-r*');
title('Logistic模型拟合图');
xlabel('年份');
ylabel('人口数(万人)');
legend('非线性最小二乘拟合预测人口数量');

%%



