%代码说明：参考如下链接的文章，基本可以实现对GDP的预测
%参考文章： https://blog.csdn.net/m0_62526778/article/details/128983299
%% 区域GDP预测
clc;clear;
% 1. 读取数据 - 请将'B.xlsx'替换为您的数据文件名，并将'data(:,2)'根据要预测的列确定
% data = readmatrix('B.xlsx');
% time_series_data = data(:,2);
time_series_data = [41383.87 45952.65 50660.20 55580.11 60359.43 65552.00 70665.71 75752.20 80827.71 85556.13 88683.21]';
% 2. 划分训练集和测试集 - 这里使用80%的数据作为训练集，您可以根据需要调整比例
train_size = round(length(time_series_data) * 0.8);
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
best_mdl = arima(best_p, best_d, best_q);
 
% 6. 拟合模型
EstMdl = estimate(best_mdl, train_data);
 
% 7. 对测试集数据后的值进行预测 - 设定预测步长
num_steps =  40; % 预测测试集之后的20天数据
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
hold on;
plot(time_series_data, 'r--*', 'LineWidth', 1);hold on
plot(train_size+1:train_size+length(test_data), test_data, 'k--o', 'LineWidth', 1); hold on% 绘制测试集数据
plot(train_size+1:train_size+num_steps, forecast, 'b--o', 'LineWidth', 1);hold on
 
 
xlim([1, length(time_series_data) + num_steps]);
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('值');
legend('实际数据', '测试集数据', '预测', 'Location', 'best');
 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);