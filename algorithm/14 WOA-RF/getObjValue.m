function objValue = getObjValue(parameter)

% Ŀ�꺯����һ������ʽ���̣�Ψһ�Ĵ����ǲ�������������������������ΪĿ�꺯����ֵ��
% ����ʾ����һ����������񣬲������׼ȷ�ʣ���С�������ʣ���Ŀ�꺯����
% ������ѵ����������Ҫ��ȡѵ�������Լ���Ӧ�ı�ǩ�������Ŀ�꺯���ڲ���ȡ���ݣ������ַ�ʽ��
% �ڸ�ʵ���У����õ����ַ�ʽ���д���
% ��1������ѵ�����ݺͱ�ǩ��ȫ�ֱ���
% ��2������load������ȡѵ�����ݺͱ�ǩ
% ��3������evalin������ȡ�������ռ��ѵ�����ݺͱ�ǩ

%%  ���������л�ȡѵ������
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');

%%  ��ȡ���Ų���
    ntree = round(parameter(1, 1));          % number of trees
    mtry  = round(parameter(1, 2));          % default is floor(sqrt(size(X,2)

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
    model = classRF_train(pv_train, tv_train, ntree, mtry);

    %  ģ��Ԥ��
    [t_sim, ~] = classRF_predict(pv_valid, model);
    
    % ��Ӧ��ֵ
    accuracy(i) = sum((t_sim == tv_valid)) / length(tv_valid);

end

%%  �Է���Ԥ���������Ϊ�Ż���Ŀ�꺯��ֵ
    if size(accuracy, 1) == 0
        objValue = 1;
    else
        objValue = 1 - mean(accuracy);
    end

end