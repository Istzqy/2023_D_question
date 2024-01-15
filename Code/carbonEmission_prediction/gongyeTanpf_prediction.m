%%
%**代码功能：用于预测工业部门的能源消费品种结构以及对应品种的碳排放因子

clc
clear all
close all
%% 工业部门能源消费品种结构预测

% 人口预测，第一问已经预测完成是，直接进行数据读取
prePopdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');
% 工业部门能源消费预测，已经预测完成，直接进行数据读取
enePre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gongye_enePredata.xlsx');
% 2010~2020工业部门消费品种结构
meitan = [7335.50	7818.72	7510.45	7956.97	8358.12	8383.95	8417.76	8262.63	7931.34	8829.81	8291.34]';
youpin = [1401.86	1256.52	1435.90	1337.25	1436.43	1440.57	1512.38	1350.40	1351.66	1290.06	1225.65]';
tianran = [444.08	520.77	677.51	769.11	820.22	923.65	956.63	1020.41	1305.00	1377.01	1666.36]';
reli = [1632.97	1701.05	1747.25	1856.94	1802.47	1846.36	1982.91	2118.22	2248.05	2252.10	2234.25]';
dianli = [3462.07	3841.64	4080.38	4405.26	4573.72	4614.14	4825.18	5049.36	5247.23	5318.40	5400.85]';
qita = [36.19	33.16	43.11	22.27	27.57	33.52	29.70	31.31	40.38	42.18	54.28]';

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
best_q = 5;
best_d = 1;
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
title('2021~2060工业消费部门煤炭消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');


 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域工业部门煤炭消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_meitan_xiaohao.xlsx',meitan_xiaohao);


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
best_p = 8;
best_q = 1;
best_d = 1;
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

%11. 区域工业部门油品消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_youpin_xiaohao.xlsx',youpin_xiaohao);

%% 天然气消费预测 
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = tianran';
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
ylabel('工业消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
tianran_2 = [tianran' G(12:21)];
figure(3)
plot(t2,G,'o',t2,tianran_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

time_series_data = tianran_2' ;
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
best_p = 7;
best_q = 1;
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
ylabel('天然气消耗预测');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
tianran_xiaohao =  [train_data' forecast']';
plot(2021:2060,tianran_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060工业消费部门天然气消耗预测');
xlabel('时间');
ylabel('万tce');
%xlim([1, length(time_series_data) + num_steps]);
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域工业部门天然气消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_tianran_xiaohao.xlsx',tianran_xiaohao);
 
%% 热力消费预测 
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = reli';
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
ylabel('热力消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
reli_2 = [reli' G(12:21)];
figure(3)
plot(t2,G,'o',t2,reli_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

time_series_data = reli_2' ;
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
best_q = 1;
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
ylabel('万tce');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
reli_xiaohao =  [train_data' forecast']';
plot(2021:2060,reli_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060工业消费部门热力消耗预测');
xlabel('时间');
ylabel('万tce');
%xlim([1, length(time_series_data) + num_steps]);
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域工业部门热力消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_reli_xiaohao.xlsx',reli_xiaohao);
 
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
best_p = 4;
best_q = 1;
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
title('2021~2060工业消费部门电力消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域工业部门电力消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_dianli_xiaohao.xlsx',dianli_xiaohao);

%% 其他消费预测
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = qita';
B=cumsum(A);  % 原始数据累加
years=10;     %预测未来年数
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
ylabel('其他能源消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
qita_2 = [qita' G(12:21)];
figure(3)
plot(t2,G,'o',t2,qita_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')

time_series_data = qita_2' ;
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
best_p = 7;
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
qita_xiaohao =  [train_data' forecast']';
plot(2021:2060,qita_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060工业消费部门其他消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域工业部门其他消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_qita_xiaohao.xlsx',qita_xiaohao);
          

%% 区域工业部门消耗预测输出
clear all;
% 人口预测，第一问已经预测完成是，直接进行数据读取
prePopdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');
% 工业部门能源消费预测，已经预测完成，直接进行数据读取
enePre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gongye_enePredata.xlsx');
% 分别读取工业部门各种能源的预测
meitan_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_meitan_xiaohao.xlsx');
youpin_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_youpin_xiaohao.xlsx');
tianran_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_tianran_xiaohao.xlsx');
reli_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_reli_xiaohao.xlsx');
dianli_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_dianli_xiaohao.xlsx');
qita_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gy_qita_xiaohao.xlsx');
total = meitan_xiaohao + youpin_xiaohao + tianran_xiaohao + reli_xiaohao + dianli_xiaohao + qita_xiaohao ;
%计算比例
for i = 1:51
    bili(i,1) = meitan_xiaohao(i)/ total(i);
    bili(i,2) = youpin_xiaohao(i)/total(i);
    bili(i,3) = tianran_xiaohao(i)/total(i);
    bili(i,4) = reli_xiaohao(i)/total(i);
    bili(i,5) = dianli_xiaohao(i)/ total(i);
    bili(i,6) = qita_xiaohao(i)/total(i);
end
% 换成实际能源品种消耗
xiaohaofenbu(:,1) = enePre .* bili(:,1);
xiaohaofenbu(:,2) = enePre .* bili(:,2);
xiaohaofenbu(:,3) = enePre .* bili(:,3);
xiaohaofenbu(:,4) = enePre .* bili(:,4);
xiaohaofenbu(:,5) = enePre .* bili(:,5);
xiaohaofenbu(:,6) = enePre .* bili(:,6);

xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gongye_anjiegou_xiaohao.xlsx',xiaohaofenbu);

%%  消费部门碳排放因子预测
% 思路：碳排放因子预测应与二次能源消耗比例相关联
% 对于工业部门:主要体现在电力、其他能源消耗比例提升对碳排放因子影响

meitan_yz = [2.961	2.632	2.664	3.130	3.302	3.091	3.086	3.166	3.256	3.348	3.383];
youpin_yz = [1.377	1.496	1.577	1.519	1.594	1.301	1.323	1.364	1.452	1.558	1.833];
tianran_yz = [1.628	1.572	1.628	1.629	1.587	1.593	1.573	1.599	1.604	1.601	1.605];
reli_yz = [2.707	2.861	2.985	2.603	2.620	2.739	2.817	2.926	3.194	3.083	3.124];
dianli_yz = [6.307	6.445	6.253	6.253	5.450	5.778	5.725	5.696	5.596	5.649	5.406];



% 对各种能源的碳排放因子进行预测，采用多元线性回归的方式
% 参与回归的变量是各种能源消费的比例
% 农林部门参与回归的三种比例是煤炭、油品与电力
meitan_bili = bili(:,1);
youpin_bili = bili(:,2);
tianran_bili = bili(:,3);
reli_bili = bili(:,4);
dianli_bili = bili(:,5);
qita_bili = bili(:,6);

%多元线性回归
len = length(meitan_yz);
pelta = ones(len,1);
x1 = [pelta, meitan_bili(1:11),youpin_bili(1:11),tianran_bili(1:11),reli_bili(1:11),dianli_bili(1:11),qita_bili(1:11)];
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x1);


%存在多重共线性问题，采用岭回归
ridge_x= [meitan_bili(1:11), youpin_bili(1:11),tianran_bili(1:11),reli_bili(1:11),dianli_bili(1:11),qita_bili(1:11)];
ridge_y1 = meitan_yz';
ridge_y2 = youpin_yz';
ridge_y3 = dianli_yz';
ridge_y4 = tianran_yz';
ridge_y5 = reli_yz';
k = 0:0.1:10;
B = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
B3 = ridge(ridge_y3,ridge_x,k,0);
B4 = ridge(ridge_y4,ridge_x,k,0);
B5 = ridge(ridge_y5,ridge_x,k,0);
for  k1 = 1:length(k)
    A=B(:,k1);
    A2 = B2(:,k1);
    A3 = B3(:,k1);
    A4 = B4(:,k1);
    A5 = B5(:,k1);
    yn= A(1)+ridge_x*A(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
    yn3= A3(1)+ridge_x*A3(2:end);
    yn4= A4(1)+ridge_x*A4(2:end);
    yn5= A5(1)+ridge_x*A5(2:end);
    wucha(k1)=sum(abs(ridge_y1-yn)./ridge_y1)/length(ridge_y1);
    wucha2(k1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
    wucha3(k1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
    wucha4(k1)=sum(abs(ridge_y4-yn4)./ridge_y4)/length(ridge_y4);
    wucha5(k1)=sum(abs(ridge_y5-yn5)./ridge_y5)/length(ridge_y5);
end
figure(5)
subplot(3,2,1)
plot(1:length(k),wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(3,2,2)
plot(1:length(k),wucha2,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(3,2,3)
plot(1:length(k),wucha3,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(3,2,4)
plot(1:length(k),wucha4,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(3,2,5)
plot(1:length(k),wucha5,'LineWidth',2)
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
index=find(wucha4==min(wucha4));
xishu4 = ridge(ridge_y4,ridge_x,k(index),0);
y_p4= xishu4(1)+ridge_x*xishu4(2:end);
index=find(wucha5==min(wucha5));
xishu5 = ridge(ridge_y5,ridge_x,k(index),0);
y_p5= xishu5(1)+ridge_x*xishu5(2:end);



figure(6)
t1 = 2010:2020;
subplot(3,2,1)
plot(t1,meitan_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(3,2,2)
plot(t1,youpin_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p2,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(3,2,3)
plot(t1,dianli_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p3,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(3,2,4)
plot(t1,tianran_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p4,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(3,2,5)
plot(t1,reli_yz,'m--o','LineWidth',1);
hold on
plot(t1,y_p5,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');


%2021~2060年各能源碳排放因子预测
ridge_x = [meitan_bili, youpin_bili,tianran_bili,reli_bili,dianli_bili,qita_bili];
meitan_yz_pre = xishu(1)+ridge_x*xishu(2:end);
youpin_yz_pre = xishu2(1)+ridge_x*xishu2(2:end);
dianli_yz_pre = xishu3(1)+ridge_x*xishu3(2:end);
tianran_yz_pre = xishu4(1)+ridge_x*xishu4(2:end);
reli_yz_pre = xishu5(1)+ridge_x*xishu5(2:end);


t2 = 2010:2060;
meitan_yz_pre(1:11) = meitan_yz;
youpin_yz_pre(1:11) = youpin_yz;
dianli_yz_pre(1:11) = dianli_yz;
tianran_yz_pre(1:11) = tianran_yz;
reli_yz_pre(1:11) = reli_yz;

figure(7)
subplot(3,2,1)
plot(t2,meitan_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测工业部门煤炭的碳排放因子','Location','northwest');
subplot(3,2,2)
plot(t2,youpin_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测工业部门油品的碳排放因子','Location','northwest');
subplot(3,2,3)
plot(t2,dianli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测工业部门电力的碳排放因子','Location','northwest');
subplot(3,2,4)
plot(t2,tianran_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测工业部门天然气的碳排放因子','Location','northwest');
subplot(3,2,5)
plot(t2,reli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测工业部门热力的碳排放因子','Location','northwest');

%% 工业部门碳排放量预测

gongye_tanpf_pre(:,1) = xiaohaofenbu(:,1).*meitan_yz_pre;
gongye_tanpf_pre(:,2) = xiaohaofenbu(:,2).*youpin_yz_pre;
gongye_tanpf_pre(:,3) = xiaohaofenbu(:,5).*dianli_yz_pre;
gongye_tanpf_pre(:,4) = xiaohaofenbu(:,3).*tianran_yz_pre;
gongye_tanpf_pre(:,5) = xiaohaofenbu(:,4).*reli_yz_pre;
gongye_tanpf_pre(:,6) = gongye_tanpf_pre(:,1)+gongye_tanpf_pre(:,2)+gongye_tanpf_pre(:,3)+gongye_tanpf_pre(:,4)+gongye_tanpf_pre(:,5) ;

figure(8)
plot(t2,gongye_tanpf_pre);
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('工业部门碳排放量预测');
legend({'煤炭碳排放','油品碳排放','电力碳排放','天然气碳排放','热力碳排放','工业部门碳排放'});

% 工业部门碳排放量预测数据输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gongye_tanpf_pre.xlsx',gongye_tanpf_pre);















