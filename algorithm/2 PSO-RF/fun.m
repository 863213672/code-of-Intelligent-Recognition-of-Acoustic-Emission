function fitness = fun(pop, p_train, t_train)

%%  ��ȡ���Ų���
n_trees = round(pop(1));
n_layer = round(pop(2));

%%  ���ݵĲ���
num_size = length(t_train);

%%  ������֤����
indices = crossvalind('Kfold', num_size, 5);

for i = 1 : 5
    
    % ��ȡ��i�����ݵ������߼�ֵ
    valid_data = (indices == i);
    
    % ȡ������ȡ��i��ѵ�����ݵ������߼�ֵ
    train_data = ~valid_data;
    
    % 1�ݲ��ԣ�4��ѵ��
    pv_train = p_train(train_data, :);
    tv_train = t_train(train_data, :);
    
    pv_valid = p_train(valid_data, :);
    tv_valid = t_train(valid_data, :);

    % ����ģ��
    model = classRF_train(pv_train, tv_train, n_trees, n_layer);

    % �������
    t_sim = classRF_predict(pv_valid, model);

    % ��Ӧ��ֵ
    error(i) = 1 - sum(t_sim == tv_valid) ./ length(tv_valid);

end

%%  ��ȡ��Ӧ��
fitness = mean(error);