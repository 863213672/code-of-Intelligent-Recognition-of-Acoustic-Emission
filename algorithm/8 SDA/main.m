%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据
res = xlsread('数据集.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
% res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）
flag_conusion = 1;                        % 标志位为1，打开混淆矩阵（要求2018版本及以上）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = 44000;         % 得到该类别的训练样本个数

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
p_test  = mapminmax('apply', P_test, ps_input);
t_train = ind2vec(T_train);
t_test  = ind2vec(T_test );

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  参数设置
f_ = size(p_train, 2);                       % 输入特征个数
hidden   = [f_, 30, 30];                     % 自动编码器的隐藏层节点个数[30, 30]
sae_lr   = 0.5;                              % 自动编码器的学习率
sae_mark = 0.5;                              % 自动编码器的噪声覆盖率（不为零时，为堆叠去噪自动编码器）
sae_act  = 'sigm';                           % 自动编码器的激活函数

opts.numepochs = 500;                        % 自动编码器的最大训练次数
opts.batchsize = M / 3;                      % 每次训练样本个数 需满足：（M / batchsize = 整数）

%%  建立堆叠自编码器
sae = saesetup(hidden);

%%  堆叠自动编码器参数设置
for i = 1 : length(hidden) - 1
    sae.ae{i}.activation_function     = sae_act;    % 激活函数
    sae.ae{i}.learningRate            = sae_lr;     % 学习率
    sae.ae{i}.inputZeroMaskedFraction = sae_mark;   % 噪声覆盖率
end

%%  训练堆叠自动编码器
sae = saetrain(sae, p_train, opts);

%%  建立深层神经网络
nn = nnsetup([hidden, num_class]);

%%  权重赋值
for i = 1 : length(hidden) - 1
    nn.W{i} = sae.ae{i}.W{1};
end

%%  训练深层神经网络
nn.activation_function  = 'sigm';    % 激活函数
nn.learningRate         = 2;         % 学习率
nn.momentum             = 0.5;       % 动量参数
nn.scaling_learningRate = 1;         % 学习率的比例因子
opts.numepochs          = 2000;      % 迭代次数
opts.batchsize          = M / 3;     % 每次训练样本个数 需满足：（M / batchsize = 整数）

%%  训练网络
[nn, loss, accu] = nntrain(nn, p_train, t_train, opts);

%%  模型预测
T_sim1 = nnpredict(nn, p_train);
T_sim2 = nnpredict(nn, p_test );

%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  损失函数曲线
figure
subplot(2, 1, 1)
plot(1 : length(accu), accu, 'r-', 'LineWidth', 1)
xlabel('迭代次数')
ylabel('准确率')
legend('训练集正确率')
title ('训练集正确率曲线')
grid
    
subplot(2, 1, 2)
plot(1 : length(loss), loss, 'b-', 'LineWidth', 1)
xlabel('迭代次数')
ylabel('损失函数')
legend('训练集损失值')
title ('训练集损失函数曲线')
grid

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