%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  添加路径
addpath("RF_Toolbox\")

%%  读取数据
res = xlsread('数据集.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）
f_ = size(res, 2) - 1;                    % 输入特征的特征数目

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  参数初始化
c1      =  4.495;            % 学习因子
c2      =  4.494;            % 学习因子
maxgen  =    50;             % 种群更新次数  
sizepop =     5;             % 种群规模
Vmax    = [ 100.0,  2.0];    % 最大速度(树的数目，树的深度)
Vmin    = [-100.0, -2.0];    % 最小速度(树的数目，树的深度)
popmax  = [  800, f_] ;      % 最大边界(树的数目，树的深度，最大深度不能超过特征数)
popmin  = [  200,  1];       % 最小边界(树的数目，树的深度，最小深度为1)

%%  种群初始化
for i = 1 : sizepop
    pop(i, :) = rand(1, length(popmax)) .* (popmax - popmin) + popmin;
    V(i, :) = rand(1, length(Vmax)) .* Vmax + 1;
    fitness(i) = fun(pop(i, :), p_train, t_train);
end

%%  个体极值和群体极值
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % 全局最佳
gbest = pop;                   % 个体最佳
fitnessgbest = fitness;        % 个体最佳适应度值
BestFit = fitnesszbest;        % 全局最佳适应度值

%%  迭代寻优
for i = 1 : maxgen
    for j = 1 : sizepop

        % 速度更新
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        
        if (sum(V(j, :) > Vmax) > 0 || sum(V(j, :) < Vmin) > 0)
            [~, vmax_index] = find(V(j, :) > Vmax);
            [~, vmin_index] = find(V(j, :) < Vmin);
            V(j, vmax_index) = Vmax(vmax_index);
            V(j, vmin_index) = Vmin(vmin_index);
        end
        
        % 种群更新
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);

        if (sum(pop(j, :) > popmax) > 0 || sum(pop(j, :) < popmin) > 0)
            [~, pmax_index] = find(pop(j, :) > popmax);
            [~, pmin_index] = find(pop(j, :) < popmin);
            pop(j, pmax_index) = popmax(pmax_index);
            pop(j, pmin_index) = popmin(pmin_index);
        end
        
        % 自适应变异
        if rand > 0.95
            pop(j, :) = rand(1, length(popmax)) .* (popmax - popmin) + 1;
        end
        
        % 适应度值
        fitness(j) = fun(pop(j, :), p_train, t_train);
    end
    
    for j = 1 : sizepop
        
        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % 群体最优更新 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

%%  提取最优参数
n_trees = round(zbest(1));
n_layer = round(zbest(2));

%%  创建模型
model = classRF_train(p_train, t_train, n_trees, n_layer);
importance = model.importance; % 特征的重要性

%%  仿真测试
[T_sim1, Vote1] = classRF_predict(p_train, model);
[T_sim2, Vote2] = classRF_predict(p_test , model);

%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  适应度曲线
figure;
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
title('适应度曲线', 'FontSize', 13);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度值', 'FontSize', 10);
xlim([1, length(BestFit)])
grid

%%  绘制特征重要性
figure
bar(model.importance)
legend('重要性')
xlabel('特征')
ylabel('重要性')

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%  混淆矩阵
if flag_conusion == 1

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
end