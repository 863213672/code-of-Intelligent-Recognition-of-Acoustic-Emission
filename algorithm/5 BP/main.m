%%  清空环境变量
clc                     % 清空命令行
clear all                  % 清空变量
close all               % 关闭开启的图窗
warning off             % 关闭报警信息
%%  导入数据
res = xlsread('数据集.xlsx');
%%  划分训练集和测试集
qq=size(res,2);
aqq=88000;
P_train = res(1: aqq, 1: qq-1)';
T_train = res(1: aqq, qq)';
M = size(P_train, 2);

P_test = res(aqq+1: end, 1: qq-1)';
T_test = res(aqq+1: end, qq)';
N = size(P_test, 2);
%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test );

%%  建立模型
net = newff(p_train, t_train, 10);

%%  设置训练参数
net.trainParam.epochs = 1000;   % 最大迭代次数
net.trainParam.goal = 1e-6;     % 目标训练误差
net.trainParam.lr = 0.01;       % 学习率

%%  训练网络
net = train(net, p_train, t_train);

%%  仿真测试
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test );

%%  数据反归一化
T_sim1 = vec2ind(t_sim1);
T_sim2 = vec2ind(t_sim2);

%%  数据排序
[T_train, index_1] = sort(T_train);
% [T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
% T_sim2 = T_sim2(index_2);

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {strcat('训练集预测结果对比：', ['准确率=' num2str(error1) '%'])};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {strcat('测试集预测结果对比：', ['准确率=' num2str(error2) '%'])};
title(string)
grid

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
