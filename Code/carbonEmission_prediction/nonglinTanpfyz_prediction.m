%%
%**代码功能：用于预测农林部门的能源消费品种结构以及对应品种的碳排放因子

clc
clear all
close all
%% 农林部门能源消费品种结构预测

% 人口预测，第一问已经预测完成是，直接进行数据读取
prePopdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');
% 农林部门能源消费预测，已经预测完成，直接进行数据读取
enePre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nonglin_enePredata.xlsx');
% 2010~2020农林部门消费品种结构
meitan = [36.19	40.24	38.72	36.65	36.18	36.05	35.02	34.64	33.66	32.38	31.76]';
youpin = [274.31	313.05	363.57	272.19	290.25	328.79	322.40	323.78	335.87	312.03	293.32]';
dianli = [34.85	40.57	46.65	53.34	57.30	64.53	76.04	82.07	88.46	94.17	98.70]';

%% 煤炭消费预测
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = meitan';
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
figure(1)
t1=2010:2020;
t2=2010:2030;
plot(t1,A,'o',t2,G,'r--*');  %原始数据与预测数据的比较
xlabel('年份');
ylabel('煤炭消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
meitan_2 = [meitan' G(12:21)];
figure(3)
plot(t2,G,'o',t2,meitan_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

time_series_data = meitan_2' ;
% 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
train_size = round(length(time_series_data)*0.8);
train_data = time_series_data(1:train_size);
test_data = time_series_data(train_size+1:end);
 
% 3. 初始化最小AIC和BIC以及最优参数 - 选择模型参数的范围（p、d、q的最大值）
max_p = 8;
max_d = 2;
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

%已求出 best_p = 4  best_q = 2  best_d = 1
best_p = 6;
best_q = 4;
best_d = 0;
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
plot(t2(1)+train_size:t2(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('GDP');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
meitan_xiaohao =  [train_data' forecast']';
plot(2021:2060,meitan_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060农林消费部门煤炭消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');


 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域农林部门煤炭消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nl_meitan_xiaohao.xlsx',meitan_xiaohao);


%% 油品消费预测 
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = youpin';
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
figure(1)
t1=2010:2020;
t2=2010:2030;
plot(t1,A,'o',t2,G,'r--*');  %原始数据与预测数据的比较
xlabel('年份');
ylabel('油品消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
youpin_2 = [youpin' G(12:21)];
figure(3)
plot(t2,G,'o',t2,youpin_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

time_series_data = youpin_2' ;
% 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
train_size = round(length(time_series_data)*0.8);
train_data = time_series_data(1:train_size);
test_data = time_series_data(train_size+1:end);
 
% 3. 初始化最小AIC和BIC以及最优参数 - 选择模型参数的范围（p、d、q的最大值）
max_p = 8;
max_d = 2;
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

%已求出 best_p = 6  best_q = 4  best_d = 2
best_p = 6;
best_q = 4;
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
plot(t2(1)+train_size:t2(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('GDP');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
youpin_xiaohao =  [train_data' forecast']';
plot(2021:2060,youpin_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060农林消费部门油品消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域农林部门油品消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nl_youpin_xiaohao.xlsx',youpin_xiaohao);


%% 电力消费预测
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = dianli';
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
figure(1)
t1=2010:2020;
t2=2010:2030;
plot(t1,A,'o',t2,G,'r--*');  %原始数据与预测数据的比较
xlabel('年份');
ylabel('电力消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
dianli_2 = [dianli' G(12:21)];
figure(3)
plot(t2,G,'o',t2,dianli_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

time_series_data = dianli_2' ;
% 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
train_size = round(length(time_series_data)*0.8);
train_data = time_series_data(1:train_size);
test_data = time_series_data(train_size+1:end);
 
% 3. 初始化最小AIC和BIC以及最优参数 - 选择模型参数的范围（p、d、q的最大值）
max_p = 8;
max_d = 2;
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

%已求出 best_p = 6  best_q = 4  best_d = 2
best_p = 8;
best_q = 3;
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
plot(t2(1)+train_size:t2(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('GDP');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
dianli_xiaohao =  [train_data' forecast']';
plot(2021:2060,dianli_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060农林消费部门电力消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域农林部门电力消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nl_dianli_xiaohao.xlsx',dianli_xiaohao);



%% 消费比例计算
meitan_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nl_meitan_xiaohao.xlsx');
youpin_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nl_youpin_xiaohao.xlsx');
dianli_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nl_dianli_xiaohao.xlsx');
total = meitan_xiaohao + youpin_xiaohao + dianli_xiaohao;
for i = 1:51
    bili(i,1) = meitan_xiaohao(i)/ total(i);
    bili(i,2) = youpin_xiaohao(i)/total(i);
    bili(i,3) = dianli_xiaohao(i)/total(i);
end



%% 区域农林部门消耗预测输出
xiaohaofenbu(:,1) = enePre .* bili(:,1);
xiaohaofenbu(:,2) = enePre .* bili(:,2);
xiaohaofenbu(:,3) = enePre .* bili(:,3);


xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nonglin_anjiegou_xiaohao.xlsx',xiaohaofenbu);


%% 消费部门碳排放因子预测
% 思路：碳排放因子预测应与二次能源消耗比例相关联
% 对于农林部门:主要体现在电力消耗比例提升对碳排放因子影响

meitan_yz = [2.664	2.664	2.664	2.664	2.664	2.664	2.664	2.664	2.664	2.664	2.664];
youpin_yz = [2.114	2.116	2.119	2.117	2.109	2.110	2.117	2.117	2.116	2.116	2.116];
dianli_yz = [6.307	6.445	6.253	6.253	5.450	5.778	5.725	5.696	5.596	5.649	5.406];

% 对各种能源的碳排放因子进行预测，采用多元线性回归的方式
% 参与回归的变量是各种能源消费的比例
% 农林部门参与回归的三种比例是煤炭、油品与电力
meitan_bili = bili(:,1);
youpin_bili = bili(:,2);
dianli_bili = bili(:,3);

%多元线性回归
len = length(meitan_yz);
pelta = ones(len,1);
x1 = [pelta, meitan_bili(1:11), youpin_bili(1:11),dianli_bili(1:11)];
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x1);


%存在多重共线性问题，采用岭回归
ridge_x= [meitan_bili(1:11), youpin_bili(1:11),dianli_bili(1:11)];
ridge_y1 = meitan_yz';
ridge_y2 = youpin_yz';
ridge_y3 = dianli_yz';
k = 0:0.1:10;
B = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
B3 = ridge(ridge_y3,ridge_x,k,0);
for  k1 = 1:length(k)
    A=B(:,k1);
    A2 = B2(:,k1);
    A3 = B3(:,k1);
    yn= A(1)+ridge_x*A(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
    yn3= A3(1)+ridge_x*A3(2:end);
    wucha(k1)=sum(abs(ridge_y1-yn)./ridge_y1)/length(ridge_y1);
    wucha2(k1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
    wucha3(k1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
end
figure(5)
subplot(3,1,1)
plot(1:length(k),wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(3,1,2)
plot(1:length(k),wucha2,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(3,1,3)
plot(1:length(k),wucha3,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')

index=find(wucha==min(wucha));
xishu = ridge(ridge_y1,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
index=find(wucha2==min(wucha2));
xishu2 = ridge(ridge_y2,ridge_x,k(index),0);
y_p2= xishu2(1)+ridge_x*xishu2(2:end);
index=find(wucha3==min(wucha3));
xishu3 = ridge(ridge_y3,ridge_x,k(index),0);
y_p3= xishu3(1)+ridge_x*xishu3(2:end);


figure(6)
t1 = 2010:2020;
subplot(3,1,1)
plot(t1,meitan_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(3,1,2)
plot(t1,youpin_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p2,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(3,1,3)
plot(t1,dianli_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p3,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');

%2021~2060年各能源碳排放因子预测
ridge_x = [meitan_bili youpin_bili dianli_bili];
meitan_yz_pre = xishu(1)+ridge_x*xishu(2:end);
youpin_yz_pre = xishu2(1)+ridge_x*xishu2(2:end);
dianli_yz_pre = xishu3(1)+ridge_x*xishu3(2:end);


t2 = 2010:2060;
meitan_yz_pre(1:11) = meitan_yz;
youpin_yz_pre(1:11) = youpin_yz;
dianli_yz_pre(1:11) = dianli_yz;
figure(7)
subplot(3,1,1)
plot(t2,meitan_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测农林部门煤炭的碳排放因子','Location','northwest');
subplot(3,1,2)
plot(t2,youpin_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测农林部门油品的碳排放因子','Location','northwest');
subplot(3,1,3)
plot(t2,dianli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测农林部门电力的碳排放因子','Location','northwest');

%% 农林部门碳排放量预测
nonglin_tanpf_pre(:,1) = xiaohaofenbu(:,1).*meitan_yz_pre;
nonglin_tanpf_pre(:,2) = xiaohaofenbu(:,2).*youpin_yz_pre;
nonglin_tanpf_pre(:,3) = xiaohaofenbu(:,3).*dianli_yz_pre;
nonglin_tanpf_pre(:,4) = nonglin_tanpf_pre(:,1)+nonglin_tanpf_pre(:,2)+nonglin_tanpf_pre(:,3);

figure(8)
plot(t2,nonglin_tanpf_pre);
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('农林部门碳排放量预测');
legend({'煤炭碳排放','油品碳排放','电力碳排放','农林部门碳排放'});

% 农林部门碳排放量预测数据输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nonglin_tanpf_pre.xlsx',nonglin_tanpf_pre);







