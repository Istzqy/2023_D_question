
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

%% 能源供应部门GDP预测灰色预测（短期预测10年内，弥补数据量不足缺陷）

syms a b;
c=[a b]';
nybmGdp = [904.65 947.43 1121.15 1065.45 1149.82 1357.63 1417.90 1526.98 1604.56 1692.70 1660.68 ];
% t1 = 2010:2020;
%A=[174,179,183,189,207,234,220.5,256,270,285];
A = nybmGdp;
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

%% 时间序列预测
%利用灰色预测得到一个新的Gdp序列（扩充10个数据进行预测）
nybmGdp_2 = [nybmGdp G(12:21)];
figure(3)
plot(t2,G,'o',t2,nybmGdp_2,'r--*')  %原始数据与预测数据的比较
xlabel('年份')
ylabel('Gdp')


time_series_data = nybmGdp_2' ;
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
%best_p = 8  best_q = 1  best_d = 2
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
figure(4);
subplot(2,1,1)
plot(t2,time_series_data, 'r--*', 'LineWidth', 1);hold on
%plot(t2(1)+train_size:t2(1)+train_size+length(test_data)-1, test_data, 'k--o', 'LineWidth', 1); hold on% 绘制测试集数据
plot(t2(1)+train_size:t2(1)+train_size+num_steps-1, forecast, 'b--*', 'LineWidth', 1);hold on
title('ARIMA 时间序列预测');
xlabel('时间');
ylabel('GDP');
legend('实际数据', '预测', 'Location', 'best');
subplot(2,1,2)
ny_gdp =  [train_data' forecast']';
plot(2021:2060,ny_gdp(12:51) , 'b--*', 'LineWidth', 1);
title('2021~2060能源供应部门GDP预测');
xlabel('时间');
ylabel('GDP(亿元)');
%xlim([1, length(time_series_data) + num_steps]);

 
% 10. 输出模型参数
disp(['最优模型参数: p = ', num2str(best_p), ', d = ', num2str(best_d), ', q = ', num2str(best_q)]);
disp(['最小 AIC: ', num2str(min_aic)]);
disp(['最小 BIC: ', num2str(min_bic)]);

%区域能源部门GDP预测输出
% xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_gdp.xlsx',ny_gdp);

%% 多元线性回归分析预测能源消耗量

% 能源供应部门供电能源消费量预测
tce_dian = [5752.10	6992.75	7339.98	7820.62	6951.05	7380.49	7786.52	8219.69	8104.87	8131.55	7491.05]';
tce_re = [ 204.84 286.73 354.09	335.18 382.10 440.86 415.01	523.98 1010.79 953.12 1003.97]';
tce_qita = [-610.67	243.37	223.05	-968.04	-1181.69 -1275.59 -1334.32 -1635.70 -2028.82 -2376.43 -2414.66]';
tce_sun = [ 454.13	450.76	457.50	353.57	500.15	474.01	337.92	377.27	370.15	378.33	372.90]';

len = length(tce_dian);
pelta = ones(len,1);
x = [pelta, prePopdata(1:11), ny_gdp(1:11)];

[b,bint,r,rint,stats]=regress(tce_dian,x,0.05);          % 95%的置信区间
[b1,bint1,r1,rint1,stats1] = regress(tce_re,x,0.05);     % 95%的置信区间
[b2,bint2,r2,rint2,stats2]=regress(tce_qita,x,0.05);     % 95%的置信区间
[b3,bint3,r3,rint3,stats3] = regress(tce_sun,x,0.05); 

tce_dian_NiHe = b(1) + b(2) .*  prePopdata(1:11) + b(3) .* ny_gdp(1:11)  ;
tce_re_NiHe = b1(1) + b1(2) .*  prePopdata(1:11) + b1(3) .* ny_gdp(1:11)  ;
tce_qita_NiHe = b2(1) + b2(2) .*  prePopdata(1:11) + b2(3) .* ny_gdp(1:11)  ;
tce_sun_NiHe = b3(1) + b3(2) .*  prePopdata(1:11) + b3(3) .* ny_gdp(1:11)  ;

figure(5);
subplot(2,2,1);
hold on;
plot(t1,tce_dian,'m--o','LineWidth',1);
plot(t1,tce_dian_NiHe,'kx-','LineWidth',1);
title('发电能源消耗');
legend({'能源消费量（万tce）','多元线性回归拟合'},'Location','northwest');
R_2 = 1 - sum( (tce_dian_NiHe - tce_dian).^2 )./ sum( (tce_dian - mean(tce_dian)).^2 );
str = num2str(R_2);
disp(['发电拟合优度为：',str]);
subplot(2,2,2);
hold on;
title('供热能源消耗');
plot(t1,tce_re,'m--o','LineWidth',1);
plot(t1,tce_re_NiHe,'kx-','LineWidth',1);
legend({'能源消费量（万tce）','多元线性回归拟合'},'Location','northwest');
R_2 = 1 - sum( (tce_re_NiHe - tce_re).^2 )./ sum( (tce_re - mean(tce_re)).^2 );
str = num2str(R_2);
disp(['供热拟合优度为：',str]);
subplot(2,2,3);
hold on;
title('其他转换');
plot(t1,tce_qita,'m--o','LineWidth',1);
plot(t1,tce_qita_NiHe,'kx-','LineWidth',1);
legend({'能源消费量（万tce）','多元线性回归拟合'},'Location','northeast');
R_2 = 1 - sum( (tce_qita_NiHe - tce_qita).^2 )./ sum( (tce_qita - mean(tce_qita)).^2 );
str = num2str(R_2);
disp(['其他转换拟合优度为：',str]);
subplot(2,2,4);
hold on;
title('损失');
plot(t1,tce_sun,'m--o','LineWidth',1);
plot(t1,tce_sun_NiHe,'kx-','LineWidth',1);
legend({'能源消费量（万tce）','多元线性回归拟合'},'Location','northwest');
R_2 = 1 - sum( (tce_sun_NiHe - tce_sun).^2 )./ sum( (tce_sun - mean(tce_sun)).^2 );
str = num2str(R_2);
disp(['损失拟合优度为：',str]);



 
% 多重共线性检验
%计算自变量的相关系数矩阵，发现两自变量相关系数高达0.98以上，因此需要进一步进行正则化处理
correlation_matrix = corrcoef(x);

% 岭回归预测
ridge_x= [prePopdata(1:11) ny_gdp(1:11)];
ridge_y = tce_dian;
ridge_y1 = tce_re;
ridge_y2 = tce_qita;
ridge_y3 = tce_sun;
k = 0:0.1:10;
B = ridge(ridge_y,ridge_x,k,0);
B1 = ridge(ridge_y1,ridge_x,k,0);
B2 = ridge(ridge_y2,ridge_x,k,0);
B3 = ridge(ridge_y3,ridge_x,k,0);
for  k1 = 1:length(k)
    A=B(:,k1);
    A1 = B1(:,k1);
    A2 = B2(:,k1);
    A3 = B3(:,k1);
    yn= A(1)+ridge_x*A(2:end);
    yn1 = A1(1)+ridge_x*A1(2:end);
    yn2= A2(1)+ridge_x*A2(2:end);
    yn3 = A3(1)+ridge_x*A3(2:end);
    wucha(k1)=sum(abs(ridge_y-yn)./ridge_y)/length(ridge_y);
    wucha1(k1)=sum(abs(ridge_y1-yn1)./ridge_y1)/length(ridge_y1);
    wucha2(k1)=sum(abs(ridge_y2-yn2)./ridge_y2)/length(ridge_y2);
    wucha3(k1)=sum(abs(ridge_y3-yn3)./ridge_y3)/length(ridge_y3);
end
figure(6)
subplot(2,2,1);
plot(1:length(k),wucha,'LineWidth',2);
subplot(2,2,2);
plot(1:length(k),wucha1,'LineWidth',2);
subplot(2,2,3);
plot(1:length(k),wucha2,'LineWidth',2);
subplot(2,2,4);
plot(1:length(k),wucha3,'LineWidth',2);
ylabel('相对误差');
xlabel('岭参数');

index=find(wucha==min(wucha));
xishu_dian = ridge(ridge_y,ridge_x,k(index),0);
index=find(wucha1==min(wucha1));
xishu_re = ridge(ridge_y1,ridge_x,k(index),0);
index=find(wucha2==min(wucha2));
xishu_qita = ridge(ridge_y2,ridge_x,k(index),0);
index=find(wucha3==min(wucha3));
xishu_sun = ridge(ridge_y3,ridge_x,k(index),0);

y_dian= xishu_dian(1)+ridge_x*xishu_dian(2:end);
y_re= xishu_re(1)+ridge_x*xishu_re(2:end);
y_qita =  xishu_qita(1)+ridge_x*xishu_qita(2:end);
y_sun =  xishu_sun(1)+ridge_x*xishu_sun(2:end);

figure(7)
subplot(2,2,1);
plot(t1,tce_dian,'m--o','LineWidth',1);
hold on
plot(t1,y_dian,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(2,2,2);
plot(t1,tce_re,'m--o','LineWidth',1);
hold on
plot(t1,y_re,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');
subplot(2,2,3);
plot(t1,tce_qita,'m--o','LineWidth',1);
hold on
plot(t1,y_qita,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northeast');
subplot(2,2,4);
plot(t1,tce_sun,'m--o','LineWidth',1);
hold on
plot(t1,y_sun,'b--*','LineWidth',1);
legend({'原始数据','岭回归预测'},'Location','northwest');






%% 2021~2060能源消费量预测
ridge_x1 = [prePopdata ny_gdp];
enePre_dian = xishu_dian(1)+ridge_x1*xishu_dian(2:end);
enePre_re = xishu_re(1)+ridge_x1*xishu_re(2:end);
enePre_qita = xishu_qita(1)+ridge_x1*xishu_qita(2:end);
enePre_sun = xishu_sun(1)+ridge_x1*xishu_sun(2:end);

t2 = 2010:2060;
enePre_dian(1:11) = tce_dian;
enePre_re(1:11) = tce_re;
enePre_qita(1:11) = tce_qita;
enePre_sun(1:11) = tce_sun;

figure(8)
subplot(2,2,1);
plot(t2,enePre_dian,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');
subplot(2,2,2);
plot(t2,enePre_re,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');
subplot(2,2,3);
plot(t2,enePre_qita,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northeast');
subplot(2,2,4);
plot(t2,enePre_sun,'B--o','LineWidth',1);
legend('岭回归预测能源消费量','Location','northwest');


total = enePre_dian + enePre_re + enePre_qita + enePre_sun;
nybm_enePredata = [enePre_dian enePre_re enePre_qita enePre_sun total];


xlswrite('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nybm_enePredata.xlsx',nybm_enePredata);



