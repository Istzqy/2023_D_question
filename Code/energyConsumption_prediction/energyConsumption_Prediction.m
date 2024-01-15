% 代码说明：能源消费量预测

%% 数据预处理
clc
clear
close all
%2010~2020年人口数据
pop =  [7869.34	8022.99	8119.81	8192.44	8281.09	8315.11	8381.47	8423.50	8446.19	8469.09	8477.26]';
%2010~2020年GDP数据
gdp = [41383.87 45952.65 50660.20 55580.11 60359.43 65552.00 70665.71 75752.20 80827.71 85556.13 88683.21]';
%2010~2020年能源消费量数据
tce = [23539.31	26860.03 27999.22 28203.10	28170.51 29033.61 29947.98 30669.89	31373.13 32227.51 31438.00]';
t1=2010:2020;
figure(1);
hold on;
plot(t1,pop,'b--o');
plot(t1,gdp,'r--o');
plot(t1,tce,'m--o','LineWidth',1);
legend({'人口数量(万人)','GDP数据(亿元)','能源消费量（万tce）'},'Location','northwest');

%% 多元线性回归分析

% 因为用的3是维拟合，则 x 应该为 3*15 的矩阵，第一列为 1 ，第二列为 x1 ，第三列为 x2 , 第四列为 x3
% 15 代表的是 样本个数
len = length(tce);
pelta = ones(len,1);
x = [pelta, pop, gdp];

[b,bint,r,rint,stats]=regress(tce,x,0.05);     % 95%的置信区间

tce_NiHe = b(1) + b(2) .* pop + b(3) .* gdp  ;

figure(2);
hold on;
plot(t1,pop,'b--o');
plot(t1,gdp,'r--o');
plot(t1,tce,'m--o','LineWidth',1);
plot(t1,tce_NiHe,'kx-','LineWidth',1);
legend({'人口数量(万人)','GDP数据(亿元)','能源消费量（万tce）','多元线性回归拟合'},'Location','northwest');
R_2 = 1 - sum( (tce_NiHe - tce).^2 )./ sum( (tce - mean(tce)).^2 );
str = num2str(R_2);
disp(['拟合优度为：',str])
 
%% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数高达0.98以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x);

%% 岭回归
%b=ridge(y,x,k,s);
%b是岭回归模型中的系数向量β=[β0，β1，β2，...，βn],β0是常数项，β1到βn是自变量x1到xn对应的系数
%y是因变量向量
%x是自变量矩阵，x=[x1,...,xn],每个xi都是列向量
%k是岭参数，岭参数不同，岭回归模型不同，要选取合适的岭参数
%s这个位置的参数只能填0或1，或者不填默认为0。0表示输出的系数β该是多少就是多少，1表示输出系数β是标准化后的


% ridge_x= [pop gdp];
% ridge_y = tce;


%先画出岭迹图，以便选取合适的岭参数
% k=0:1e-5:5e-3;    %岭参数
% b2=ridge(ridge_y,ridge_x,k,0); %回归系数
% 
% %岭迹图，一般选取开始平稳的“拐点”处的k值
% figure(3)
% plot(k,b2(1,:));
% hold on
% plot(k,b2(2,:));
% hold on
% plot(k,b2(3,:));
% plot(k,b2);
% xlabel('k');
% ylabel('β');
% title('岭迹');
% legend('pop','gdp','popgdp');



% %每个k对应的残差平方和的图，要选取会使残差平方和小的k值
% knum=size(b2,2);
% %sse=zeros(knum);
% y_gu=zeros(11,knum);
% for j=1:knum
%     t2=0;
%     for i=1:11
%         y_gu(i,j)=b2(1,j)*1+b2(2,j)*pop(i)+b2(3,j)*gdp(i);
%         t2=(y_gu(i,j)-ridge_y(i))^2+t2;
%     end
%         sse(j)=t2;
% end
% figure(4)
% plot(k,sse)
% xlabel('k')
% ylabel('SSE')
% title('残差平方和与k的关系图')

% %下面均是岭参数k=2情况
%  b3=ridge(ridge_y,ridge_x,2,0);%2是岭参数。最后一个位置的参数，是否输出标准化系数，0否1是
% y_gu=[[ones(18,1),x(：,:)]*b]1;%用岭回归模型求出来的函数估计值（向量）

%% 岭回归预测
ridge_x= [pop gdp];
ridge_y = tce;
k = 0:0.1:10;
B = ridge(ridge_y,ridge_x,k,0);
for  k1 = 1:length(k)
    A=B(:,k1);
    yn= A(1)+ridge_x*A(2:end);
    wucha(k1)=sum(abs(ridge_y-yn)./ridge_y)/length(ridge_y);
end
figure(5)
subplot(2,1,1);
plot(k(1:50),wucha(1:50),'LineWidth',2);
ylabel('相对误差');
xlabel('岭参数');

index=find(wucha==min(wucha));
xishu = ridge(ridge_y,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
subplot(2,1,2);
plot(t1,tce,'m--o','LineWidth',2);
hold on
plot(t1,y_p,'b--*','LineWidth',2);
legend({'原始数据','岭回归预测'},'Location','northwest');
xlabel('年份');
ylabel('能源消耗量(万tce)');


%% 2021~2060能源消费量预测
popPre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');
preGdp = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\Economy_prediction\preGdpData.xlsx');
ridge_x1 = [popPre preGdp];
enePre = xishu(1)+ridge_x1*xishu(2:end);
t2 = 2010:2060;
enePre(1:11) = tce;
figure(7)
plot(t2,enePre,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\energyConsumption_prediction\enePredata.xlsx',enePre);









