
% *碳排放量预测* 
%% 数据预处理
clc
clear all
close all
enePre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\energyConsumption_prediction\enePredata.xlsx');
yiConsum = [ 345.36	393.87	448.95	362.18	383.72	429.37	433.46	440.49	457.99	438.58	423.78 ]';
nybmConsum = [5800.39 7973.61 8374.63 7541.33 6651.60 7019.77 7205.12 7485.24 7457.00 7086.56 6453.25]';
gybmConsum = [14312.68 15171.86	15494.59 16347.81 17018.53	17242.19 17724.55 17832.32 18123.66	19109.55 18872.74]';
jtbmConsum = [1398.26 1494.74 1618.15 1743.67 1915.88 2019.29 2083.59 2187.96 2324.65 2482.93 2484.47]';
jzbmConsum = [534.58 620.83	690.81 738.21 728.32 756.83	794.94	861.24 976.14 1039.46 1023.15]';
jmxfConsum = [1148.05 1205.13 1372.09 1469.91 1472.45 1566.16 1706.31 1862.64 2033.69 2070.43 2180.60]';
totalConsum = yiConsum + nybmConsum + gybmConsum + jtbmConsum + jzbmConsum + jmxfConsum ;
%每年能源消费在各部门的分布
xffenbu = ones(6,11);
%计算能源消费的平均分布（假设各部门的能源消费分布）
avefen = ones(6,1);
for i = 1:11
    xffenbu(1,i) = yiConsum(i)/totalConsum(i);
    xffenbu(2,i) = nybmConsum(i)/totalConsum(i);
    xffenbu(3,i) = gybmConsum(i)/totalConsum(i);
    xffenbu(4,i) = jtbmConsum(i)/totalConsum(i);
    xffenbu(5,i) = jzbmConsum(i)/totalConsum(i);
    xffenbu(6,i) = jmxfConsum(i)/totalConsum(i);
    avefen(1) = xffenbu(1,i) + avefen(1);
    avefen(2) = xffenbu(2,i) + avefen(2);
    avefen(3) = xffenbu(3,i) + avefen(3);
    avefen(4) = xffenbu(4,i) + avefen(4);
    avefen(5) = xffenbu(5,i) + avefen(5);
    avefen(6) = xffenbu(6,i) + avefen(6);
end
avefen = avefen / 11;

figure(1);
t1 = 2010 : 2020 ;
plot(t1,xffenbu(1,:));
hold on;
plot(t1,xffenbu(2,:));
hold on;
plot(t1,xffenbu(3,:));
hold on;
plot(t1,xffenbu(4,:));
hold on
plot(t1,xffenbu(5,:));
hold on
plot(t1,xffenbu(6,:));
legend('农林','能源供应','工业','交通','建筑','居民');
ylabel('比例系数');
xlabel('年份');




%% 各部门能源消费量预测
%思路:
%    1、分别对各部门的能源消费量进行回归预测，参与回归预测的变量为对应部门的产值与GDP
%    2、由各部分能源消费量预测，计算出各部门的消费占比
%    3、利用第一问中预测的能源消费总量乘以消费占比，得到2010~2060年的各部门能源消费预测
yiGdp = [2409.24 2736.86 3057.82 3228.54 3358.61 3636.08 3690.61 3568.54 3591.61 3726.61 3916.81]';
nybmGdp = [904.65 947.43 1121.15 1065.45 1149.82 1357.63 1417.90 1526.98 1604.56 1692.70 1660.68 ]';
gybmGdp = [20948.95	22792.53 24491.76 26232.68	27757.72 29342.78 30595.12	32987.35 34929.18 36037.45	36522.55]';
jtbmGdp = [1767.22 1988.43	2199.51	2233.94	2378.93	2240.39	2316.43	2420.17	2570.68	2749.08	2761.51]';
jzbmGdp = [15353.81	17487.40 19789.96 22819.50	25714.36 28975.11 32645.64	35249.16 38131.69	41350.30 43821.67]';

prePopdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');

%农林部门能源消费量预测
len = length(yiConsum);
pelta = ones(len,1);
x = [pelta, prePopdata(1:11), yiGdp];

[b,bint,r,rint,stats]=regress(yiConsum,x,0.05);     % 95%的置信区间

yi_tce_NiHe = b(1) + b(2) .* prePopdata(1:11) + b(3) .* yiGdp  ;

figure(2);
hold on;
plot(t1,yiConsum,'m--o','LineWidth',1);
plot(t1,yi_tce_NiHe,'kx-','LineWidth',1);
legend({'原始能源消费量（万tce）','多元线性回归拟合'},'Location','northwest');
R_2 = 1 - sum( (yi_tce_NiHe - yiConsum).^2 )./ sum( (yiConsum - mean(yiConsum)).^2 );
str = num2str(R_2);
disp(['拟合优度为：',str])

%%% 多重共线性校验，计算自变量的相关系数矩阵，发现两自变量相关系数高达0.98以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x);

%%% 岭回归预测
ridge_x= [prePopdata(1:11) yiGdp];
ridge_y = yiConsum;
k = 0:0.1:10;
B = ridge(ridge_y,ridge_x,k,0);
for  k1 = 1:length(k)
    A=B(:,k1);
    yn= A(1)+ridge_x*A(2:end);
    wucha(k1)=sum(abs(ridge_y-yn)./ridge_y)/length(ridge_y);
end
figure(5)
plot(1:length(k),wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')

index=find(wucha==min(wucha));
xishu = ridge(ridge_y,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
figure(6)
plot(t1,yiConsum,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');

%% 2021~2060能源消费量预测
preGdp = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\Economy_prediction\preGdpData.xlsx');
ridge_x1 = [prePopdata preGdp];
enePre = xishu(1)+ridge_x1*xishu(2:end);
t2 = 2010:2060;
enePre(1:11) = tce;
figure(7)
plot(t2,enePre,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\energyConsumption_prediction\enePredata.xlsx',enePre);





