%%
%*****代码功能：用于预测能源部门的能源消费品种结构以及对应品种的碳排放因子******%

clc
clear all
close all
%% 能源部门能源消费品种结构预测

% 2010~2020能源部门消费品种结构
meitan = [10530.19	13115.37	13610.82	13202.63	12130.49	12426.87	12992.04	12759.39	12683.30	12204.50	11969.34]';
youpin = [172.95	140.74	214.55	183.97	235.18	182.21	204.88	163.32	221.18	280.59	277.74]';
tianran = [373.73	520.16	611.67	601.83	543.31	847.59	876.80	1663.34	1819.16	1774.45	1368.52]';
reli = [-1669.89 -1727.73 -1764.35 -1882.89 -1824.49 -1865.89 -1989.92 -2124.77 -2255.88 -2260.63 -2244.51]';
dianli = [-3731.81 -4218.09 -4493.99 -4822.18 -4725.06 -4869.85 -5164.95 -5286.40 -5346.75 -5261.69 -5217.57]';
qita = [125.22	143.15	195.92	257.97	292.18	298.84	286.27	310.35	335.99	349.35	299.73]';


%% 预测能源部门煤炭、油品、天然气、热力、电力、其他能源消耗量（灰色预测+时间序列）

%****** 煤炭预测********%
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
best_p = 4;
best_q = 2;
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
title('2021~2060能源消费部门煤炭消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');


 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域能源部门煤炭消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_meitan_xiaohao.xlsx',meitan_xiaohao);


%****** 油品预测********%
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

%已求出 best_p = 4  best_q = 2  best_d = 1
best_p = 4;
best_q = 5;
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
title('2021~2060能源消费部门油品消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');


 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域能源部门油品消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_youpin_xiaohao.xlsx',youpin_xiaohao);




%****** 天然气预测********%
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
ylabel('天然气消费');

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

%已求出 best_p = 4  best_q = 2  best_d = 1
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
ylabel('GDP');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
tianran_xiaohao =  [train_data' forecast']';
plot(2021:2060,tianran_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060能源消费部门天然气消耗预测');
xlabel('时间');
ylabel('GDP(亿元)');


% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域能源部门天然气消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_tianran_xiaohao.xlsx',tianran_xiaohao);



%****** 热力预测********%
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = abs(reli)';
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
ylabel('热力产出');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
reli_2 = [abs(reli)' G(12:21)];
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

%已求出 best_p = 4  best_q = 2  best_d = 1
best_p = 8;
best_q = 0;
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
reli_chanchu =  [train_data' forecast']';
plot(2021:2060,reli_chanchu(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060能源消费部门热力产出预测');
xlabel('时间');
ylabel('万tce');


% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域能源部门热力阐述预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_reli_chanchu.xlsx',reli_chanchu*-1);






%****** 电力预测********%
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = abs(dianli)';
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
ylabel('电力产出(万tce)');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
dianli_2 = [abs(dianli)' G(12:21)];
figure(3)
plot(t2,G,'o',t2,dianli_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('万tce')

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

%已求出 best_p = 4  best_q = 2  best_d = 1
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
dianli_chanchu =  [train_data' forecast']';
plot(2021:2060,dianli_chanchu(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060能源消费部门热力产出预测');
xlabel('时间');
ylabel('GDP(亿元)');


% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域能源部门电力产出预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_dianli_chanchu.xlsx',dianli_chanchu*-1);




%****** 其他能源预测********%
%*******灰色预测********%（短期预测10年内，弥补数据量不足缺陷）
syms a b;
c=[a b]';
A = qita';
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
ylabel('其他能源消费');

%******* 时间序列预测 ********%
% %利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
qita_2 = [qita' G(12:21)];
figure(3)
plot(t2,G,'o',t2,qita_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('万tce')

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

%已求出 best_p = 4  best_q = 2  best_d = 1
best_p = 7;
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
ylabel('万tce');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
qita_xiaohao =  [train_data' forecast']';
plot(2021:2060,qita_xiaohao(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060能源消费部门其他能源消耗预测');
xlabel('时间');
ylabel('万tce');


% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%11. 区域部门其他消耗预测输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_qita_xiaohao.xlsx',qita_xiaohao);


%% 能源部门消费品种比例预测

% 人口预测，第一问已经预测完成是，直接进行数据读取
prePopdata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\population_prediction\prePopData.xlsx');
% 能源部门能源消费预测，已经预测完成，直接进行数据读取
nybm_enePredata = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nybm_enePredata.xlsx');
% 能源部门消费品种预测
meitan_xiaohao =  xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_meitan_xiaohao.xlsx');
youpin_xiaohao =  xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_youpin_xiaohao.xlsx');
tianran_xiaohao =  xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_tianran_xiaohao.xlsx');
reli_chanchu = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_reli_chanchu.xlsx');
dianli_chanchu = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_dianli_chanchu.xlsx');
qita_xiaohao = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_qita_xiaohao.xlsx');

% 以煤炭消耗量为基准，计算能源供应部门能源供应结构预测，得到其比例
% x1 + x2 + x3 -abs(x4) - abs(x5) + x6 = X 
% x1 + k1*x1 +  k2*x1 - k3*x1 - k4*x1 + k5*x1 = X
% x1 = X/(1+k1+k2-k3-k4+k5)
% x2 = k1 * x1
% x3 = k2 * x1
% x4 = -k3 * x1
% x5 = -k4 * x1
% x6 = k5 * x1

k1 = ones(51,1);
k2 = ones(51,1);
k3 = ones(51,1);
k4 = ones(51,1);
k5 = ones(51,1);

k1 = youpin_xiaohao./meitan_xiaohao;
k2 = tianran_xiaohao./meitan_xiaohao;
k3 = abs(reli_chanchu)./meitan_xiaohao;
k4 = abs(dianli_chanchu)./meitan_xiaohao;
k5 = qita_xiaohao./meitan_xiaohao;



%% 分别计算能源供应部门中，发电，供热，其他转换，损失的 能源消费量

%% *发电碳排放计算* 
% DESCRIPTIVE TEXT
fadian_xiaohao = nybm_enePredata(:,1);

%根据表格，发电消耗了煤，油，气，热与其他共5种
%利用比例换算出对应的能源消耗
fadian_mei_xiaohao = fadian_xiaohao ./(1+k1+k2-k3+k5) ;
fadian_you_xiaohao = k1 .* fadian_mei_xiaohao;
fadian_qi_xiaohao = k2 .* fadian_mei_xiaohao;
fadian_re_chanchu = -k3.*fadian_mei_xiaohao;
fadian_qita_xiaohao = k5.*fadian_mei_xiaohao;

fadian_mei_tpfyz = [2.730	2.657	2.656	2.759	2.656	2.784	2.789	2.810	2.878	2.938	2.951];
fadian_you_tpfyz = [2.111	2.274	2.141	2.097	2.116	2.085	2.106	2.080	1.971	2.058	2.049];
fadian_qi_tpfyz = [1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628];
%缺值失进行均值补充
fadian_reli_tpfyz = [2.906	2.916	2.942	2.967	2.998	3.107	3.190	3.278	3.745	3.662	3.705];
fadian_qita_tpfyz = [0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000];

%碳排放因多元线性回归，回归变量为k1,k2,k3,k5
%多元线性回归
len = length(fadian_mei_tpfyz);
pelta = ones(len,1);
x1 = [pelta, k1(1:11), k2(1:11),k3(1:11),k5(1:11)];
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x1);

%存在多重共线性问题，采用岭回归
ridge_x= [k1(1:11), k2(1:11),k3(1:11),k5(1:11)];
ridge_y1 = fadian_mei_tpfyz';
ridge_y2 = fadian_you_tpfyz';
ridge_y3 = fadian_qi_tpfyz';
ridge_y4 = fadian_reli_tpfyz';
k = 0:0.1:10;
B = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
B3 = ridge(ridge_y3,ridge_x,k,0);
B4 = ridge(ridge_y4,ridge_x,k,0);
for  p1 = 1:length(k)
    A=B(:,p1);
    A2 = B2(:,p1);
    A3 = B3(:,p1);
    A4 = B4(:,p1);
    yn= A(1)+ridge_x*A(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
    yn3= A3(1)+ridge_x*A3(2:end);
    yn4= A4(1)+ridge_x*A4(2:end);
    wucha(p1)=sum(abs(ridge_y1-yn)./ridge_y1)/length(ridge_y1);
    wucha2(p1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
    wucha3(p1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
    wucha4(p1)=sum(abs(ridge_y4-yn4)./ridge_y4)/length(ridge_y4);
end
figure(5)
subplot(2,2,1)
plot(1:length(k),wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(2,2,2)
plot(1:length(k),wucha2,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(2,2,3)
plot(1:length(k),wucha3,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(2,2,4)
plot(1:length(k),wucha4,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')


index=find(wucha==min(wucha));
xishu = ridge(ridge_y1,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
index=find(wucha2==min(wucha2));
xishu2 = ridge(ridge_y2,ridge_x,k(index),0);
y_p2= xishu2(1)+ridge_x*xishu2(2:end);
index=find(wucha3==min(wucha3));
xishu3 = ridge(ridge_y3,ridge_x,k(index(1)),0);
y_p3= xishu3(1)+ridge_x*xishu3(2:end);
index=find(wucha4==min(wucha4));
xishu4 = ridge(ridge_y4,ridge_x,k(index),0);
y_p4= xishu4(1)+ridge_x*xishu4(2:end);


figure(6)
t1 = 2010:2020;
subplot(2,2,1)
plot(t1,fadian_mei_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'煤原始数据','岭回归预测'});
subplot(2,2,2)
plot(t1,fadian_you_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p2,'b--*','LineWidth',1);
legend({'油品原始数据','岭回归预测'},'Location','northwest');
subplot(2,2,3)
plot(t1,fadian_qi_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p3,'b--*','LineWidth',1);
legend({'天然气原始数据','岭回归预测'},'Location','northwest');
subplot(2,2,4)
plot(t1,fadian_reli_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p4,'b--*','LineWidth',1);
legend({'热力原始数据','岭回归预测'},'Location','northwest');

%2021~2060年能源部门发电量部分碳排放因子预测
ridge_x = [k1 k2 k3 k5];
fadian_meitan_yz_pre = xishu(1)+ridge_x*xishu(2:end);
fadian_youpin_yz_pre = xishu2(1)+ridge_x*xishu2(2:end);
fadian_qi_yz_pre = xishu3(1)+ridge_x*xishu3(2:end);
fadian_reli_yz_pre = xishu4(1)+ridge_x*xishu4(2:end);

t2 = 2010:2060;
fadian_meitan_yz_pre(1:11) = fadian_mei_tpfyz;
fadian_youpin_yz_pre(1:11) = fadian_you_tpfyz;
fadian_qi_yz_pre(1:11) = fadian_qi_tpfyz;
fadian_reli_yz_pre(1:11) = fadian_reli_tpfyz;

figure(7)
subplot(2,2,1)
plot(t2,fadian_meitan_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分煤炭的碳排放因子','Location','northwest');
subplot(2,2,2)
plot(t2,fadian_youpin_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分油品的碳排放因子','Location','northwest');
subplot(2,2,3)
plot(t2,fadian_qi_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分天然气的碳排放因子','Location','northwest');
subplot(2,2,4)
plot(t2,fadian_reli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分热力的碳排放因子','Location','northwest');


% 能源部门发电量部分碳排放量预测
ny_fadian_tanpf_pre(:,1) = fadian_mei_xiaohao .* fadian_meitan_yz_pre;
ny_fadian_tanpf_pre(:,2) = fadian_you_xiaohao .* fadian_youpin_yz_pre;
ny_fadian_tanpf_pre(:,3) = fadian_qi_xiaohao.*fadian_qi_yz_pre;
ny_fadian_tanpf_pre(:,4) = fadian_re_chanchu.*fadian_reli_yz_pre;
ny_fadian_tanpf_pre(:,5) = ny_fadian_tanpf_pre(:,1)+ny_fadian_tanpf_pre(:,2)+ny_fadian_tanpf_pre(:,3)+ny_fadian_tanpf_pre(:,4);


figure(8)
plot(t2,ny_fadian_tanpf_pre(:,5));
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('能源部门发电碳排放量预测');
legend({'能源部门发电碳排放'});

% 能源部门发电碳排放预测数据输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_fadian_tanpf_pre.xlsx',ny_fadian_tanpf_pre);


%% *供热碳排放计算* 
% DESCRIPTIVE TEXT
gongre_xiaohao = nybm_enePredata(:,2);

%根据表格，发电消耗了煤，油，气与其他共4种
%利用比例换算出对应的能源消耗
gongre_mei_xiaohao = gongre_xiaohao ./(1+k1+k2-k3+k5) ;
gongre_you_xiaohao = k1 .* gongre_mei_xiaohao;
gongre_qi_xiaohao = k2 .* gongre_mei_xiaohao;
% fadian_re_chanchu = -k3.*fadian_mei_xiaohao;
gongre_qita_xiaohao = k5.*gongre_mei_xiaohao;

gongre_mei_tpfyz = [2.783	2.656	2.657	2.680	2.834	2.824	2.818	2.824	2.791	2.788	2.766];
gongre_you_tpfyz = [2.002	2.273	2.078	1.994	1.967	1.956	1.959	1.926	1.934	1.922	2.021];
gongre_qi_tpfyz = [1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628];
%缺值失进行均值补充
%fadian_reli_tpfyz = [2.906	2.916	2.942	2.967	2.998	3.107	3.190	3.278	3.745	3.662	3.705];
gongre_qita_tpfyz = [0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000];

%碳排放因多元线性回归，回归变量为k1,k2,k5
%多元线性回归
len = length(gongre_mei_tpfyz);
pelta = ones(len,1);
x1 = [pelta, k1(1:11), k2(1:11),k5(1:11)];
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x1);

%存在多重共线性问题，采用岭回归
ridge_x= [k1(1:11), k2(1:11),k5(1:11)];
ridge_y1 = gongre_mei_tpfyz';
ridge_y2 = gongre_you_tpfyz';
ridge_y3 = gongre_qi_tpfyz';
%ridge_y4 = fadian_reli_tpfyz';
k = 0:0.1:10;
B = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
B3 = ridge(ridge_y3,ridge_x,k,0);
%B4 = ridge(ridge_y4,ridge_x,k,0);
for  p1 = 1:length(k)
    A=B(:,p1);
    A2 = B2(:,p1);
    A3 = B3(:,p1);
 %   A4 = B4(:,p1);
    yn= A(1)+ridge_x*A(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
    yn3= A3(1)+ridge_x*A3(2:end);
%    yn4= A4(1)+ridge_x*A4(2:end);
    wucha(p1)=sum(abs(ridge_y1-yn)./ridge_y1)/length(ridge_y1);
    wucha2(p1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
    wucha3(p1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
%    wucha4(p1)=sum(abs(ridge_y4-yn4)./ridge_y4)/length(ridge_y4);
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
xishu3 = ridge(ridge_y3,ridge_x,k(index(1)),0);
y_p3= xishu3(1)+ridge_x*xishu3(2:end);
% index=find(wucha4==min(wucha4));
% xishu4 = ridge(ridge_y4,ridge_x,k(index),0);
% y_p4= xishu4(1)+ridge_x*xishu4(2:end);


figure(6)
t1 = 2010:2020;
subplot(3,1,1)
plot(t1,gongre_mei_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'煤原始数据','岭回归预测'});
subplot(3,1,2)
plot(t1,gongre_you_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p2,'b--*','LineWidth',1);
legend({'油品原始数据','岭回归预测'},'Location','northwest');
subplot(3,1,3)
plot(t1,gongre_qi_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p3,'b--*','LineWidth',1);
legend({'天然气原始数据','岭回归预测'},'Location','northwest');


%2021~2060年能源部门发电量部分碳排放因子预测
ridge_x = [k1 k2 k5];
gongre_meitan_yz_pre = xishu(1)+ridge_x*xishu(2:end);
gongre_youpin_yz_pre = xishu2(1)+ridge_x*xishu2(2:end);
gongre_qi_yz_pre = xishu3(1)+ridge_x*xishu3(2:end);
% fadian_reli_yz_pre = xishu4(1)+ridge_x*xishu4(2:end);

t2 = 2010:2060;
gongre_meitan_yz_pre(1:11) = gongre_mei_tpfyz;
gongre_youpin_yz_pre(1:11) = gongre_you_tpfyz;
gongre_qi_yz_pre(1:11) = gongre_qi_tpfyz;
% fadian_reli_yz_pre(1:11) = fadian_reli_tpfyz;

figure(7)
subplot(3,1,1)
plot(t2,gongre_meitan_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门供热部分煤炭的碳排放因子','Location','northwest');
subplot(3,1,2)
plot(t2,gongre_youpin_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门供热部分油品的碳排放因子','Location','northwest');
subplot(3,1,3)
plot(t2,gongre_qi_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门供热部分天然气的碳排放因子','Location','northwest');



% 能源部门供热量部分碳排放量预测
ny_gongre_tanpf_pre(:,1) = gongre_mei_xiaohao .* gongre_meitan_yz_pre;
ny_gongre_tanpf_pre(:,2) = gongre_you_xiaohao .* gongre_youpin_yz_pre;
ny_gongre_tanpf_pre(:,3) = gongre_qi_xiaohao.* gongre_qi_yz_pre;
%ny_fadian_tanpf_pre(:,4) = abs(fadian_re_chanchu).*fadian_reli_yz_pre;
ny_gongre_tanpf_pre(:,4) = ny_gongre_tanpf_pre(:,1)+ny_gongre_tanpf_pre(:,2)+ny_gongre_tanpf_pre(:,3);


figure(8)
plot(t2,ny_gongre_tanpf_pre(:,4));
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('能源部门供热碳排放量预测');
legend({'能源部门供热碳排放'});

% 能源部门供热碳排放预测数据输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_gongre_tanpf_pre.xlsx',ny_gongre_tanpf_pre);


%% *其他转换碳排放计算* 
% DESCRIPTIVE TEXT
qitazh_xiaohao = nybm_enePredata(:,3);

%根据表格，发电消耗了煤，油，气，热与其他共5种
%利用比例换算出对应的能源消耗
qitazh_mei_xiaohao = qitazh_xiaohao ./(1+k1+k2-k3+k5) ;
qitazh_you_xiaohao = k1 .* qitazh_mei_xiaohao;
% fadian_qi_xiaohao = k2 .* fadian_mei_xiaohao;
qitazh_re_chanchu = -k3.*qitazh_mei_xiaohao;
%fadian_qita_xiaohao = k5.*fadian_mei_xiaohao;

qitazh_mei_tpfyz = [8.137	0.165	-4.646	8.864	8.042	7.885	7.858	7.320	7.333	7.260	7.277];
qitazh_you_tpfyz = [1.222	1.250	1.448	1.515	1.688	1.722	1.893	2.037	2.483	2.356	2.088];
%fadian_qi_tpfyz = [1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628];
%缺值失进行均值补充
qitazh_reli_tpfyz = [2.906	2.906	2.906	2.906	2.906	2.906	2.906	2.906	2.906	2.906	2.906];
%fadian_qita_tpfyz = [0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000];

%碳排放因多元线性回归，回归变量为k1,k3
%多元线性回归
len = length(qitazh_mei_tpfyz);
pelta = ones(len,1);
x1 = [pelta, k1(1:11), k3(1:11)];
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x1);

%存在多重共线性问题，采用岭回归
ridge_x= [k1(1:11), k3(1:11)];
ridge_y1 = qitazh_mei_tpfyz';
ridge_y2 = qitazh_you_tpfyz';
%ridge_y3 = fadian_qi_tpfyz';
ridge_y4 = qitazh_reli_tpfyz';
k = 0:0.1:10;
B = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
%B3 = ridge(ridge_y3,ridge_x,k,0);
B4 = ridge(ridge_y4,ridge_x,k,0);
for  p1 = 1:length(k)
    A=B(:,p1);
    A2 = B2(:,p1);
%    A3 = B3(:,p1);
    A4 = B4(:,p1);
    yn= A(1)+ridge_x*A(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
%    yn3= A3(1)+ridge_x*A3(2:end);
    yn4= A4(1)+ridge_x*A4(2:end);
    wucha(p1)=sum(abs(ridge_y1-yn)./ridge_y1)/length(ridge_y1);
    wucha2(p1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
%    wucha3(p1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
    wucha4(p1)=sum(abs(ridge_y4-yn4)./ridge_y4)/length(ridge_y4);
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
% subplot(2,2,3)
% plot(1:length(k),wucha3,'LineWidth',2)
% ylabel('相对误差')
% xlabel('岭参数')
subplot(3,1,3)
plot(1:length(k),wucha4,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')


index=find(wucha==min(wucha));
xishu = ridge(ridge_y1,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
index=find(wucha2==min(wucha2));
xishu2 = ridge(ridge_y2,ridge_x,k(index),0);
y_p2= xishu2(1)+ridge_x*xishu2(2:end);
% index=find(wucha3==min(wucha3));
% xishu3 = ridge(ridge_y3,ridge_x,k(index(1)),0);
% y_p3= xishu3(1)+ridge_x*xishu3(2:end);
index=find(wucha4==min(wucha4));
xishu4 = ridge(ridge_y4,ridge_x,k(index(1)),0);
y_p4= xishu4(1)+ridge_x*xishu4(2:end);


figure(6)
t1 = 2010:2020;
subplot(3,1,1)
plot(t1,qitazh_mei_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'煤原始数据','岭回归预测'});
subplot(3,1,2)
plot(t1,qitazh_you_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p2,'b--*','LineWidth',1);
legend({'油品原始数据','岭回归预测'},'Location','northwest');
subplot(3,1,3)
plot(t1,qitazh_reli_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p4,'b--*','LineWidth',1);
legend({'热力原始数据','岭回归预测'},'Location','northwest');

%2021~2060年能源部门发电量部分碳排放因子预测
ridge_x = [k1 k3];
qitazh_meitan_yz_pre = xishu(1)+ridge_x*xishu(2:end);
qitazh_youpin_yz_pre = xishu2(1)+ridge_x*xishu2(2:end);
%fadian_qi_yz_pre = xishu3(1)+ridge_x*xishu3(2:end);
qitazh_reli_yz_pre = xishu4(1)+ridge_x*xishu4(2:end);

t2 = 2010:2060;
qitazh_meitan_yz_pre(1:11) = qitazh_mei_tpfyz;
qitazh_youpin_yz_pre(1:11) = qitazh_you_tpfyz;
%fadian_qi_yz_pre(1:11) = fadian_qi_tpfyz;
qitazh_reli_yz_pre(1:11) = qitazh_reli_tpfyz;

figure(7)
subplot(3,1,1)
plot(t2,qitazh_meitan_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分煤炭的碳排放因子','Location','northwest');
subplot(3,1,2)
plot(t2,qitazh_youpin_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分油品的碳排放因子','Location','northwest');
subplot(3,1,3)
plot(t2,qitazh_reli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门发电部分热力的碳排放因子','Location','northwest');


% 能源部门发电量部分碳排放量预测
ny_qitazh_tanpf_pre(:,1) = qitazh_mei_xiaohao .* qitazh_meitan_yz_pre;
ny_qitazh_tanpf_pre(:,2) = qitazh_you_xiaohao .* qitazh_youpin_yz_pre;
ny_qitazh_tanpf_pre(:,3) = qitazh_re_chanchu.*qitazh_reli_yz_pre;
ny_qitazh_tanpf_pre(:,4) = ny_qitazh_tanpf_pre(:,1)+ny_qitazh_tanpf_pre(:,2)+ny_qitazh_tanpf_pre(:,3);
figure(8)
plot(t2,ny_qitazh_tanpf_pre(:,4));
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('能源部门碳排放量预测');
legend({'能源部门其他转换碳排放'});

% 能源部门其他转换碳排放预测数据输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_qitazh_tanpf_pre.xlsx',ny_qitazh_tanpf_pre);


%% *损失碳排放计算* 
% DESCRIPTIVE TEXT
sunshi_xiaohao = nybm_enePredata(:,4);

%根据表格，发电消耗了煤，油，气，热与其他共5种
%利用比例换算出对应的能源消耗
sunshi_mei_xiaohao = sunshi_xiaohao ./(1+k1+k2-k3+k5) ;
%qitazh_you_xiaohao = k1 .* qitazh_mei_xiaohao;
% fadian_qi_xiaohao = k2 .* fadian_mei_xiaohao;
sunshi_re_chanchu = -k3.*sunshi_mei_xiaohao;
sunshi_dian_chanchu = -k4.*sunshi_mei_xiaohao;
%fadian_qita_xiaohao = k5.*fadian_mei_xiaohao;

%qitazh_mei_tpfyz = [8.137	0.165	-4.646	8.864	8.042	7.885	7.858	7.320	7.333	7.260	7.277];
%qitazh_you_tpfyz = [1.222	1.250	1.448	1.515	1.688	1.722	1.893	2.037	2.483	2.356	2.088];
%fadian_qi_tpfyz = [1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628	1.628];
%缺值失进行均值补充
sunshi_reli_tpfyz = [3.072	3.031	3.157	3.118	3.312	3.399	3.309	3.400	3.919	3.827	3.873];
sunshi_dianli_tpfyz = [5.869	5.995	5.864	6.028	5.227	5.554	5.515	5.490	5.430	5.479	5.246];
%fadian_qita_tpfyz = [0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000	0.000];

%碳排放因多元线性回归，回归变量为k1,k3
%多元线性回归
len = length(sunshi_reli_tpfyz);
pelta = ones(len,1);
x1 = [pelta, k1(1:11),k2(1:11), k3(1:11),k4(1:11),k5(1:11)];
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x1);

%存在多重共线性问题，采用岭回归
ridge_x= [k1(1:11),k2(1:11), k3(1:11),k4(1:11),k5(1:11)];
ridge_y1 = sunshi_reli_tpfyz';
ridge_y2 = sunshi_dianli_tpfyz';
%ridge_y3 = fadian_qi_tpfyz';
%ridge_y4 = qitazh_reli_tpfyz';
k = 0:0.1:10;
B = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
%B3 = ridge(ridge_y3,ridge_x,k,0);
%B4 = ridge(ridge_y4,ridge_x,k,0);
for  p1 = 1:length(k)
    A=B(:,p1);
    A2 = B2(:,p1);
%    A3 = B3(:,p1);
%    A4 = B4(:,p1);
    yn= A(1)+ridge_x*A(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
%    yn3= A3(1)+ridge_x*A3(2:end);
%    yn4= A4(1)+ridge_x*A4(2:end);
    wucha(p1)=sum(abs(ridge_y1-yn)./ridge_y1)/length(ridge_y1);
    wucha2(p1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
%    wucha3(p1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
%    wucha4(p1)=sum(abs(ridge_y4-yn4)./ridge_y4)/length(ridge_y4);
end
figure(5)
subplot(2,1,1)
plot(1:length(k),wucha,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
subplot(2,1,2)
plot(1:length(k),wucha2,'LineWidth',2)
ylabel('相对误差')
xlabel('岭参数')
% subplot(2,2,3)
% plot(1:length(k),wucha3,'LineWidth',2)
% ylabel('相对误差')
% xlabel('岭参数')
% subplot(3,1,3)
% plot(1:length(k),wucha4,'LineWidth',2)
% ylabel('相对误差')
% xlabel('岭参数')


index=find(wucha==min(wucha));
xishu = ridge(ridge_y1,ridge_x,k(index),0);
y_p= xishu(1)+ridge_x*xishu(2:end);
index=find(wucha2==min(wucha2));
xishu2 = ridge(ridge_y2,ridge_x,k(index),0);
y_p2= xishu2(1)+ridge_x*xishu2(2:end);
% index=find(wucha3==min(wucha3));
% xishu3 = ridge(ridge_y3,ridge_x,k(index(1)),0);
% y_p3= xishu3(1)+ridge_x*xishu3(2:end);
% index=find(wucha4==min(wucha4));
% xishu4 = ridge(ridge_y4,ridge_x,k(index(1)),0);
% y_p4= xishu4(1)+ridge_x*xishu4(2:end);


figure(6)
t1 = 2010:2020;
subplot(2,1,1)
plot(t1,sunshi_reli_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p,'b--*','LineWidth',1);
legend({'损失热力原始数据','岭回归预测'});
subplot(2,1,2)
plot(t1,sunshi_dianli_tpfyz,'m--o','LineWidth',1);
hold on
plot(t1,y_p2,'b--*','LineWidth',1);
legend({'损失电力原始数据','岭回归预测'},'Location','northwest');
% subplot(3,1,3)
% plot(t1,qitazh_reli_tpfyz,'m--o','LineWidth',1);
% hold on
% plot(t1,y_p4,'b--*','LineWidth',1);
% legend({'热力原始数据','岭回归预测'},'Location','northwest');

%2021~2060年能源部门发电量部分碳排放因子预测
ridge_x = [k1 k2 k3 k4 k5];
sunshi_reli_yz_pre = xishu(1)+ridge_x*xishu(2:end);
sunshi_dianli_yz_pre = xishu2(1)+ridge_x*xishu2(2:end);
%fadian_qi_yz_pre = xishu3(1)+ridge_x*xishu3(2:end);
% qitazh_reli_yz_pre = xishu4(1)+ridge_x*xishu4(2:end);

t2 = 2010:2060;
sunshi_reli_yz_pre(1:11) = sunshi_reli_tpfyz;
sunshi_dianli_yz_pre(1:11) = sunshi_dianli_tpfyz;
%fadian_qi_yz_pre(1:11) = fadian_qi_tpfyz;
% qitazh_reli_yz_pre(1:11) = qitazh_reli_tpfyz;

figure(7)
subplot(2,1,1)
plot(t2,sunshi_reli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门损失部分热力的碳排放因子','Location','northwest');
subplot(2,1,2)
plot(t2,sunshi_dianli_yz_pre,'B--o','LineWidth',1);
legend('岭回归预测能源部门损失部分电力的碳排放因子','Location','northwest');



% 能源部门发电量部分碳排放量预测
ny_sunshi_tanpf_pre(:,1) = sunshi_re_chanchu .* sunshi_reli_yz_pre;
ny_sunshi_tanpf_pre(:,2) = sunshi_dian_chanchu .* sunshi_dianli_yz_pre;
ny_sunshi_tanpf_pre(:,3) = ny_sunshi_tanpf_pre(:,1)+ny_sunshi_tanpf_pre(:,2);
figure(8)
plot(t2,ny_sunshi_tanpf_pre(:,3));
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('能源部门碳排放量预测');
legend({'能源部门损失碳排放'});

% 能源部门损失碳排放预测数据输出
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_sunshi_tanpf_pre.xlsx',ny_sunshi_tanpf_pre);


%% 能源部门总碳排放预测
clear all

ny_fadian_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_fadian_tanpf_pre.xlsx');
ny_gongre_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_gongre_tanpf_pre.xlsx');
ny_qitazh_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_qitazh_tanpf_pre.xlsx');
ny_sunshi_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_sunshi_tanpf_pre.xlsx');

ny_tanpf_pre = ny_fadian_tanpf_pre(:,5) + ny_gongre_tanpf_pre(:,4) + ny_qitazh_tanpf_pre(:,4) + ny_sunshi_tanpf_pre(:,3) ;


figure(9)
t2 = 2010:2060;
plot(t2,ny_tanpf_pre);
xlabel('年份');
ylabel('碳排放量(万tco2)');
title('能源部门碳排放量预测');
legend({'能源部门碳排放(tco2)'});

xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_tanpf_pre.xlsx',ny_tanpf_pre);











