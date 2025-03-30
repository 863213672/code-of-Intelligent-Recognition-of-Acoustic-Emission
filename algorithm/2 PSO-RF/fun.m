function fitness = fun(pop, p_train, t_train)

%%  提取最优参数
n_trees = round(pop(1));
n_layer = round(pop(2));

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
    model = classRF_train(pv_train, tv_train, n_trees, n_layer);

    % 仿真测试
    t_sim = classRF_predict(pv_valid, model);

    % 适应度值
    error(i) = 1 - sum(t_sim == tv_valid) ./ length(tv_valid);

end

%%  获取适应度
fitness = mean(error);