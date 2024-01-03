%%
%区域经济增长预测
%代码说明：预测效果不佳。但是ADF与KPSS平稳性检验是OK的
%% 原始GDP数据绘制
clc
clear
close all
% 2010~2020区域生产总值（亿元）
t1 = 1:11;
Eco=[41383.87 45952.65 50660.20	55580.11 60359.43 65552.00 70665.71	75752.20 80827.71 85556.13	88683.21];
figure(1)
plot(t1+2009,Eco,'r--o');
xlabel('年份');
ylabel('GDP(亿元)');
hold on

% 2010~2020区域生产总值（亿元）,数据量扩充
for i = 1:length(Eco)
    if(i<11)
        Eco_ex(i*2-1) = Eco(i);
        Eco_ex(i*2) = (Eco(i)+Eco(i+1))/2;
    else
        Eco_ex(i*2-1) = Eco(i);
    end
end
t2= 1:0.5:11;
plot(t2+2009,Eco_ex,'b--*');
legend('原始数据','扩充数据');

%% 原始数据平稳性检验
%原始数据平稳性检验
[Ecod_h_adf, Ecod_p_adf,Ecod_sta_adf , Ecod_c_adf]  = adftest(Eco);
[Ecod_h_kpss, Ecod_p_kpss,Ecod_sta_kpss , Ecod_c_kpss]  = kpsstest(Eco);

%原始数据一阶差分平稳性检验
Ecod1 = diff(Eco,1);
%adf检验会报错，因为数据量太小，只有是10个
%[Ecod1_h_adf, Ecod1_p_adf,Ecod1_sta_adf , Ecod1_c_adf] = adftest(Ecod1);
[Ecod1_h_kpss, Ecod1_p_kpss,Ecod1_sta_kpss , Ecod1_c_kpss] = kpsstest(Ecod1);
figure(2)
plot(Ecod1,'r--*');
legend('一阶差分');

%原始数据二阶差分平稳性检验
Ecod2 =  diff(Eco,2);
figure(3)
plot(Ecod2,'b--*');
legend('二阶差分');
%Ecod2_h_adf = adftest(Ecod2);
[Ecod2_h_kpss, Ecod2_p_kpss,Ecod2_sta_kpss , Ecod2_c_kpss] = kpsstest(Ecod2);

%% 扩充数据平稳性检验
[Eco_exd_h_adf, Eco_exd_p_adf,Eco_exd_sta_adf , Eco_exd_c_adf]  = adftest(Eco_ex);
[Eco_exd_h_kpss, Eco_exd_p_kpss,Eco_exd_sta_kpss , Eco_exd_c_kpss]  = kpsstest(Eco_ex);

%原始数据一阶差分平稳性检验
Eco_exd1 = diff(Eco_ex,1);
%adf检验会报错，因为数据量太小，只有是10个
[Eco_exd1_h_adf, Eco_exd1_p_adf,Eco_exd1_sta_adf , Eco_exd1_c_adf]= adftest(Eco_exd1);
[Eco_exd1_h_kpss, Eco_exd1_p_kpss,Eco_exd1_sta_kpss , Eco_exd1_c_kpss]  = kpsstest(Eco_exd1);
figure(4)
plot(Eco_exd1,'r--*');
legend('扩充数据一阶差分');

%原始数据二阶差分平稳性检验
Eco_exd2 =  diff(Eco_ex,2);
figure(5)
plot(Eco_exd2,'b--*');
legend('扩充数据二阶差分');
%Ecod2_h_adf = adftest(Ecod2);
[Eco_exd2_h_adf, Eco_exd2_p_adf,Eco_exd2_sta_adf , Eco_exd2_c_adf]= adftest(Eco_exd2);
[Eco_exd2_h_kpss, Eco_exd2_p_kpss,Eco_exd2_sta_kpss , Eco_exd2_c_kpss]  = kpsstest(Eco_exd2);

%结论：经过对数据进行扩充，经过二阶差分，经过ADF与KPSS检验数据为平稳的

%% 确定AR阶数p 与 MA阶数q

figure
autocorr(Eco_exd2);
title('Autocorrelation');
figure
parcorr(Eco_exd2);
title('Partial Autocorrelation');


figure
autocorr(Ecod2);
title('原始 Autocorrelation');
figure
parcorr(Ecod2);
title('原始 Partial Autocorrelation');


figure
autocorr(diff(Eco,3));
title('原始 Autocorrelation');
figure
parcorr(diff(Eco,3));
title('原始 Partial Autocorrelation');


%% 

% 导入数据
% data = readtable('GDP_data.csv'); % 假设GDP数据存储在名为GDP_data.csv的文件中
% gdp = data.GDP;

% 数据处理
% 这里假设你已经处理过数据的格式和缺失值等问题，并得到了一个向量gdp

% 拟合模型
Eco = Eco(:);
model = arima(0, 0, 5); % 使用ARMA(2,1)模型，你可以根据自己的数据特点选择合适的模型
estimated_model = estimate(model, Eco);

% 模型评估
[residuals, ~] = infer(estimated_model, Eco);
mse = mean(residuals.^2);

% 预测未来趋势
forecast_horizon = 5; % 预测的时间范围，假设为5年
[forecasted_gdp, ~] = forecast(estimated_model, forecast_horizon);

% 可视化
figure;
plot(Eco, 'b', 'LineWidth', 1.5);
hold on;
plot(length(Eco) + (1:forecast_horizon), forecasted_gdp, 'r--', 'LineWidth', 1.5);
xlabel('年份');
ylabel('GDP');
title('GDP预测');
legend('历史GDP', '预测GDP');



