%代码说明：参考如下链接的文章，基本可以实现对GDP的预测
%参考文章： https://blog.csdn.net/m0_62526778/article/details/128983299
%% 区域GDP预测
clc;clear;
% 1. 读取数据 - 请将'B.xlsx'替换为您的数据文件名，并将'data(:,2)'根据要预测的列确定
% data = readmatrix('B.xlsx');
% time_series_data = data(:,2);
time_series_data = [41383.87 45952.65 50660.20 55580.11 60359.43 65552.00 70665.71 75752.20 80827.71 85556.13 88683.21]';
t1 = 2010:2020;
% 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
train_size = round(length(time_series_data) * 0.7);
train_data = time_series_data(1:train_size);
test_data = time_series_data(train_size+1:end);
 
% 3. 初始化最小AIC和BIC以及最优参数 - 选择模型参数的范围（p、d、q的最大值）
max_p = 5;
max_d = 2;
max_q = 5;
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
 
% 5. 使用最优参数创建ARIMA模型
best_p = 5;
best_d = 2;
best_q = 0;
best_mdl = arima(best_p, best_d, best_q);
 
% 6. 拟合模型
EstMdl = estimate(best_mdl, train_data);
 
% 7. 对测试集数据后的值进行预测 - 设定预测步长
num_steps =  43; % 预测测试集之后的20天数据
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
figure(1);
hold on;
plot(t1,time_series_data, 'r--*', 'LineWidth', 1);hold on
plot(t1(1)+train_size:t1(1)+train_size+length(test_data)-1, test_data, 'k--o', 'LineWidth', 1); hold on% 绘制测试集数据
plot(t1(1)+train_size:t1(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
 
 
%xlim([1, length(time_series_data) + num_steps]);
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('GDP');
legend('实际数据', '测试集数据', '预测', 'Location', 'best');

figure(2)
hold on;
plot(t1(1:8),time_series_data(1:8), 'r--*', 'LineWidth', 2);hold on
plot(t1(1)+train_size:t1(1)+train_size+length(test_data)-1, test_data, 'g--o', 'LineWidth', 2); hold on% 绘制测试集数据
plot(t1(1)+train_size:t1(1)+train_size+2, forecast(1:3), 'b--*', 'LineWidth', 2);hold on
% title('ARIMA 时间序列预测');
xlabel('年份');
ylabel('GDP(亿元)');
legend('训练集数据', '测试集数据', '测试集预测结果', 'Location', 'best');

 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%% 先进行灰色预测
clc;clear;
syms a b;
c=[a b]';
% t1 = 2010:2020;
%A=[174,179,183,189,207,234,220.5,256,270,285];
time_series_data = [41383.87 45952.65 50660.20 55580.11 60359.43 65552.00 70665.71 75752.20 80827.71 85556.13 88683.21]';
A = time_series_data';
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
t2=2010:2030;
plot(t2,G,'G--o')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')



%% 时间序列预测
%利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
% totalGdp_2 = [time_series_data' G(12:21)];
% figure(3)
% plot(t2,G,'o',t2,totalGdp_2,'r--*')  %原始数据与预测数据的比较
% xlabel('年份')
% ylabel('Gdp')
% 
% 
% time_series_data = totalGdp_2' ;
% % 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
% train_size = round(length(time_series_data)*0.8);
% train_data = time_series_data(1:train_size);
% test_data = time_series_data(train_size+1:end);
%  
% % 3. 初始化最小AIC和BIC以及最优参数 - 选择模型参数的范围（p、d、q的最大值）
% max_p = 8;
% max_d = 3;
% max_q = 8;
% min_aic = Inf;
% min_bic = Inf;
% best_p = 0;
% best_d = 0;
% best_q = 0;
%  
% % 4. 循环遍历不同的p, d, q值，尝试拟合ARIMA模型，并计算AIC和BIC
% for p = 0:max_p
%     for d = 0:max_d
%         for q = 0:max_q
%             % 创建ARIMA模型
%             Mdl = arima(p, d, q);
%  
%             % 拟合模型，并计算AIC和BIC
%             try
%                 [EstMdl,~,logL] = estimate(Mdl, train_data, 'Display', 'off');
%                 [aic, bic] = aicbic(logL, p + q + 1, length(train_data));
%             catch
%                 continue;
%             end
%  
%             % 更新最优参数
%             if bic < min_bic
%                 min_aic = aic;
%                 min_bic = bic;
%                 best_p = p;
%                 best_d = d;
%                 best_q = q;
%             end
%         end
%     end
% end
% 
% %已求出 best_p = 4  best_q = 2  best_d = 1
% % best_p = 4;
% % best_q = 2;
% % best_d = 1;
% % 5. 使用最优参数创建ARIMA模型
% best_mdl = arima(best_p, best_d, best_q);
%  
% % 6. 拟合模型
% EstMdl = estimate(best_mdl, train_data);
%  
% % 7. 对测试集数据后的值进行预测 - 设定预测步长
% num_steps =  34; % 预测测试集之后的35天数据
% [forecast,forecast_RMSE] = forecast(EstMdl, num_steps, 'Y0', train_data);
%  
% % 计算 95% 置信区间
% z = norminv(0.975);
% forecast_CI = [forecast - z * forecast_RMSE, forecast + z * forecast_RMSE];
%  
%  
% % 8. 输出预测结果
% disp(['预测结果（', num2str(num_steps), '个步长）:']);
% disp(forecast);
% disp(['预测置信区间（', num2str(num_steps), '个步长）:']);
% disp(forecast_CI);
%  
% % 9. 可视化预测结果
% figure;
% subplot(2,1,1)
% plot(t2,time_series_data, 'r--*', 'LineWidth', 1);hold on
% %plot(t2(1)+train_size:t2(1)+train_size+length(test_data)-1, test_data, 'k--o', 'LineWidth', 1); hold on% 绘制测试集数据
% plot(t2(1)+train_size:t2(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
% title('ARIMA 时间序列预测');
% xlabel('时间');
% ylabel('GDP');
% legend('实际数据', '预测', 'Location', 'best');
% subplot(2,1,2)
% preGdp =  [train_data' forecast']';
% plot(2021:2060,preGdp(12:51) , 'b--*', 'LineWidth', 1);
% title('2021~2060区域GDP预测');
% xlabel('时间');
% ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);


%%
%11.将GDP预测数据输出
preGdp = ones(51,1);
%前11个为原始数据
preGdp(1:11) =  time_series_data;
%后40个为2021~2060年的预测数据
preGdp(12:51) =  forecast(4:43);
xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\Economy_prediction\preGdpData.xlsx',preGdp);
% xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\Economy_prediction\preGdpData2.xlsx',preGdp);




