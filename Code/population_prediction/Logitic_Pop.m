%population.m函数文件
function g = Logitic_Pop(x,t)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
g = x(1)./(1+(x(1)/7869.34-1)*exp(-x(2)*t));  %这里的公式代入的是3.9，也就是初始数据，根据自己的初值进行修改
end
