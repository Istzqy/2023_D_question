
clear all;
clc;

%%
% 碳排放量预测数据输出
yuanshiTanpaifang = [56360.052	65193.342	67502.613	66749.376	64853.276	66074.810	68526.125	70451.557	71502.003	74096.331	72633.324]';
nonglin_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\nonglin_tanpf_pre.xlsx');
ny_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\ny_tanpf_pre.xlsx');
gongye_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\gongye_tanpf_pre.xlsx');
jiaotong_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\jiaotong_tanpf_pre.xlsx');
jianzhu_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\jianzhu_tanpf_pre.xlsx');
jumin_tanpf_pre = xlsread('D:\FPGA_MATLAB_Learning\数学建模\结课报告\D题\Code\carbonEmission_prediction\jumin_tanpf_pre.xlsx');

quyuTotalTanpaifang = nonglin_tanpf_pre(:,4)+ ny_tanpf_pre + gongye_tanpf_pre(:,6) + jiaotong_tanpf_pre(:,6) + jianzhu_tanpf_pre(:,6) + jumin_tanpf_pre(:,6);
delta_max = max(quyuTotalTanpaifang(1:11)) - max(yuanshiTanpaifang) ;
quyuTotalTanpaifang = quyuTotalTanpaifang -delta_max;
quyuTotalTanpaifang(1:11) = yuanshiTanpaifang;
figure(1);
t1 = 2010:2060;
subplot(4,2,1)
plot(t1,nonglin_tanpf_pre(:,4),'b--*','LineWidth',1);
title('区域农林总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');
subplot(4,2,2)
plot(t1,ny_tanpf_pre,'b--*','LineWidth',1);
title('区域能源总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');
subplot(4,2,3)
plot(t1,gongye_tanpf_pre(:,6),'b--*','LineWidth',1);
title('区域工业总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');
subplot(4,2,4)
plot(t1,jiaotong_tanpf_pre(:,6) ,'b--*','LineWidth',1);
title('区域交通总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');
subplot(4,2,5)
plot(t1,jianzhu_tanpf_pre(:,6) ,'b--*','LineWidth',1);
title('区域建筑总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');
subplot(4,2,6)
plot(t1,jumin_tanpf_pre(:,6),'b--*','LineWidth',1);
title('区域居民总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');
subplot(4,2,[7,8])
plot(t1,quyuTotalTanpaifang,'b--*','LineWidth',1);
title('区域居民总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');

figure(2)
plot(t1,quyuTotalTanpaifang);
title('区域总碳排放预测');
xlabel('年份');
ylabel('碳排放(tCO2)');

