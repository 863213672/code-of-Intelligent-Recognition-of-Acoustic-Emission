function objValue = getObjValue(parameter)

% 目标函数是一个非显式过程，唯一的传参是参数（或参数向量），输出参数为目标函数的值，
% 由于示例是一个多分类任务，采用最大化准确率（最小化错误率）的目标函数。
% 由于在训练过程中需要读取训练数据以及对应的标签，因此在目标函数内部读取数据，有三种方式：
% 在该实例中，采用第三种方式进行处理。
% （1）定义训练数据和标签的全局变量
% （2）利用load函数读取训练数据和标签
% （3）利用evalin函数读取主函数空间的训练数据和标签

%%  从主函数中获取训练数据
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');

%%  获取最优参数
    ntree = round(parameter(1, 1));          % number of trees
    mtry  = round(parameter(1, 2));          % default is floor(sqrt(size(X,2)

%%  数据的参数
    num_size = length(t_train);

%%  交叉验证程序
    indices = crossvalind('Kfold', num_size, 5);

for i = 1 : 5
    
    % 获取第i份数据的索引逻辑值
    valid_data = (indices == i);
    
    % 取反，获取第i份训练数据的索引逻辑值
    train_data = ~valid_data;
    
    % 1份测试，4份训练
    pv_train = p_train(train_data, :);
    tv_train = t_train(train_data, :);
    
    pv_valid = p_train(valid_data, :);
    tv_valid = t_train(valid_data, :);
    
    % 建立模型
    model = classRF_train(pv_train, tv_train, ntree, mtry);

    %  模型预测
    [t_sim, ~] = classRF_predict(pv_valid, model);
    
    % 适应度值
    accuracy(i) = sum((t_sim == tv_valid)) / length(tv_valid);

end

%%  以分类预测错误率作为优化的目标函数值
    if size(accuracy, 1) == 0
        objValue = 1;
    else
        objValue = 1 - mean(accuracy);
    end

end