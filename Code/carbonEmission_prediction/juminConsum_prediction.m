
%% 各部门能源消费量预测
%思路:
%    1、分别对各部门的能源消费量进行回归预测，参与回归预测的变量为对应部门的产值与GDP
%    2、由各部分能源消费量预测，计算出各部门的消费占比
%    3、利用第一问中预测的能源消费总量乘以消费占比，得到2010~2060年的各部门能源消费预测
clc
clear all
close all
%% 人口预测，第一问已经预测完成是，直接进行数据读取
prePopdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');

%% 区域GDP预测，第一问已经完成，直接读取
preGdpdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\Economy_prediction\preGdpData.xlsx');

%% 多元线性回归分析预测居民能源消费量


tce = [1148.05	1205.13	1372.09	1469.91	1472.45	1566.16	1706.31	1862.64	2033.69	2070.43	2180.60]';
len = length(tce);
pelta = ones(len,1);
x = [pelta, prePopdata(1:11), preGdpdata(1:11)];

[b,bint,r,rint,stats]=regress(tce,x,0.05);     % 95%的置信区间

tce_NiHe = b(1) + b(2) .*  prePopdata(1:11) + b(3) .* preGdpdata(1:11)  ;

figure(5);
t1 = 2010:2020;
hold on;
% plot(t1,prePopdata(1:11),'b--o');
% plot(t1,gy_gdp(1:11),'r--o');
plot(t1,tce,'m--o','LineWidth',1);
plot(t1,tce_NiHe,'kx-','LineWidth',1);
legend({'能源消费量（万tce）','多元线性回归拟合'},'Location','northwest');
R_2 = 1 - sum( (tce_NiHe - tce).^2 )./ sum( (tce - mean(tce)).^2 );
str = num2str(R_2);
disp(['拟合优度为：',str]);
 
%% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数高达0.9573以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x);

%% 岭回归预测能源消费量（解决多元线性回归的多重共线性问题）
ridge_x= [prePopdata(1:11) preGdpdata(1:11)];
ridge_y = tce;
k = 0:0.1:10;
B = ridge(ridge_y,ridge_x,k,0);
for  k1 = 1:length(k)
    A=B(:,k1);
    yn= A(1)+ridge_x*A(2:end);
    wucha(k1)=sum(abs(ridge_y-yn)./ridge_y)/length(ridge_y);
end
figure(6)
plot(1:length(k),wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')

index=find(wucha==min(wucha));
xishu = ridge(ridge_y,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
figure(7)
plot(t1,tce,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');



%% 2021~2060能源消费量预测
ridge_x1 = [prePopdata preGdpdata];
enePre = xishu(1)+ridge_x1*xishu(2:end);
t2 = 2010:2060;
enePre(1:11) = tce;
figure(8)
plot(t2,enePre,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');
% 居民能源消费量预测
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\jumin_enePredata.xlsx',enePre);
























