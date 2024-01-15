
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

%% 建筑部门GDP预测灰色预测（短期预测10年内，弥补数据量不足缺陷）

syms a b;
c=[a b]';
jzbmGdp = [15353.81	17487.40 19789.96 22819.50	25714.36 28975.11 32645.64	35249.16 38131.69 41350.30 43821.67];
% t1 = 2010:2020;
%A=[174,179,183,189,207,234,220.5,256,270,285];
A = jzbmGdp;
B=cumsum(A);  % 原始数据累加
years=10; %预测未来年数
n=length(A);
for i=1:(n-1)
    C(i)=(B(i)+B(i+1))/2;  % 生成累加矩阵
end
% 计算待定参数的值
D=A;D(1)=[];
D=D';
E=[-C;ones(1,n-1)];
c=inv(E*E')*E*D;
c=c';
a=c(1);b=c(2);
% 预测后续数据
F=[];F(1)=A(1);
for i=2:(n+years)
    F(i)=(A(1)-b/a)/exp(a*(i-1))+b/a ;
end
G=[];G(1)=A(1);
for i=2:(n+years)
    G(i)=F(i)-F(i-1); %得到预测出来的数据
end 
disp(['预测数据为：',num2str(G)]);
%%模型检验
%计算残差
H=G(1:11);
epsilon=A-H;
%计算相对误差q
q=abs(epsilon./A);
Q=mean(q);
disp(['平均相对误差Q检验：',num2str(Q)]);
%方差比C检验
C=std(epsilon,1)/std(A,1);
disp(['方差比C检验：',num2str(C)]);
%小误差概率P检验
S1=std(A,1);
temp=find(abs(epsilon-mean(epsilon))<0.6745*S1);
P=length(temp)/n;
disp(['小误差概率P检验：',num2str(P)]);
%绘图
figure(2)
t1=2010:2020;
t2=2010:2030;
plot(t1,A,'o',t2,G,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

%% 时间序列预测（利用灰色预测得到的数据进一步进行时间序列预测）
%利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
jzbmGdp_2 = [jzbmGdp G(12:21)];
figure(3)
plot(t2,G,'o',t2,jzbmGdp_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')


time_series_data = jzbmGdp_2' ;
% 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
train_size = round(length(time_series_data)*0.8);
train_data = time_series_data(1:train_size);
test_data = time_series_data(train_size+1:end);
 
% 3. 初始化最小AIC和BIC以及最优参数 - 选择模型参数的范围（p、d、q的最大值）
max_p = 8;
max_d = 3;
max_q = 8;
min_aic = Inf;
min_bic = Inf;
best_p = 0;
best_d = 0;
best_q = 0;
 
% 4. 循环遍历不同的p, d, q值，尝试拟合ARIMA模型，并计算AIC和BIC
for p = 0:max_p
    for d = 0:max_d
        for q = 0:max_q
            % 创建ARIMA模型
            Mdl = arima(p, d, q);
 
            % 拟合模型，并计算AIC和BIC
            try
                [EstMdl,~,logL] = estimate(Mdl, train_data, 'Display', 'off');
                [aic, bic] = aicbic(logL, p + q + 1, length(train_data));
            catch
                continue;
            end
 
            % 更新最优参数
            if bic < min_bic
                min_aic = aic;
                min_bic = bic;
                best_p = p;
                best_d = d;
                best_q = q;
            end
        end
    end
end
%best_p = 1  best_q = 0  best_d = 2
best_p = 1;
best_q = 0;
best_d = 2;
% 5. 使用最优参数创建ARIMA模型
best_mdl = arima(best_p, best_d, best_q);
 
% 6. 拟合模型
EstMdl = estimate(best_mdl, train_data);
 
% 7. 对测试集数据后的值进行预测 - 设定预测步长
num_steps =  34; % 预测测试集之后的35天数据
[forecast,forecast_RMSE] = forecast(EstMdl, num_steps, 'Y0', train_data);
 
% 计算 95% 置信区间
z = norminv(0.975);
forecast_CI = [forecast - z * forecast_RMSE, forecast + z * forecast_RMSE];
 
 
% 8. 输出预测结果
disp(['预测结果（', num2str(num_steps), '个步长）:']);
disp(forecast);
disp(['预测置信区间（', num2str(num_steps), '个步长）:']);
disp(forecast_CI);
 
% 9. 可视化预测结果
figure;
subplot(2,1,1)
plot(t2,time_series_data, 'r--*', 'LineWidth', 1);hold on
%plot(t2(1)+train_size:t2(1)+train_size+length(test_data)-1, test_data, 'k--o', 'LineWidth', 1); hold on% 绘制测试集数据
plot(t2(1)+train_size:t2(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('GDP');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
jianzhu_gdp =  [train_data' forecast']';
plot(2021:2060,jianzhu_gdp(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060建筑消费部门GDP预测');
xlabel('时间');
ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);

 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域建筑部门GDP预测输出
% xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\jianzhu_gdp.xlsx',jianzhu_gdp);


%% 多元线性回归分析预测能源消费量

% 因为用的3是维拟合，则 x 应该为 3*15 的矩阵，第一列为 1 ，第二列为 x1 ，第三列为 x2 , 第四列为 x3
% 15 代表的是 样本个数
tce = [534.58 620.83	690.81 738.21 728.32 756.83	794.94	861.24 976.14 1039.46 1023.15]';
len = length(tce);
pelta = ones(len,1);
x = [pelta, prePopdata(1:11), jianzhu_gdp(1:11)];

[b,bint,r,rint,stats]=regress(tce,x,0.05);     % 95%的置信区间

tce_NiHe = b(1) + b(2) .*  prePopdata(1:11) + b(3) .* jianzhu_gdp(1:11)  ;

figure(5);
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
%计算自变量的相关系数矩阵，发现两自变量相关系数高达0.98以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x);

%% 岭回归预测能源消费量（解决多元线性回归的多重共线性问题）
ridge_x= [prePopdata(1:11) jianzhu_gdp(1:11)];
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
ridge_x1 = [prePopdata jianzhu_gdp];
enePre = xishu(1)+ridge_x1*xishu(2:end);
t2 = 2010:2060;
enePre(1:11) = tce;
figure(8)
plot(t2,enePre,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');
% 建筑部门能源消费量预测
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\jianzhu_enePredata.xlsx',enePre);
























