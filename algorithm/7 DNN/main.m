%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��ȡ����
res = xlsread('���ݼ�.xlsx');

%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_dim = size(res, 2) - 1;               % ����ά��
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.7;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = 44000;                    % �õ���ͬ�����������
    mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % ѵ��������
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % ѵ�������

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % ���Լ�����
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % ���Լ����
end

%%  ����ת��
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  �õ�ѵ�����Ͳ�����������
M = size(P_train, 2);
N = size(P_test , 2);

%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

%%  ����ƽ��
p_train =  double(reshape(P_train, num_dim, 1, 1, M));
p_test  =  double(reshape(P_test , num_dim, 1, 1, N));

%%  ��������ṹ
layers = [
 imageInputLayer([num_dim, 1, 1])     % ����� �������ݹ�ģ[num_dim, 1, 1]
 
 fullyConnectedLayer(6)               % ȫ���Ӳ�
 reluLayer                            % Relu�����
 
 fullyConnectedLayer(4)               % ȫ���Ӳ�
 reluLayer                            % Relu�����

 fullyConnectedLayer(4)               % ȫ���Ӳ�
 reluLayer                            % Relu�����

 fullyConnectedLayer(num_class)       % ȫ���Ӳ㣨������� 
 softmaxLayer                         % ��ʧ������
 classificationLayer];                % �����

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MiniBatchSize', 100, ...              % ����С,ÿ��ѵ���������� 100
    'MaxEpochs', 500, ...                  % ���ѵ������ 500
    'InitialLearnRate', 2e-3, ...          % ��ʼѧϰ��Ϊ 0.002
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'Plots', 'training-progress', ...      % ��������
    'Verbose', false);
%%  ѵ��ģ��
net = trainNetwork(p_train, t_train, layers, options);

%%  Ԥ��ģ��
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

%%  ����һ��
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  �����������ͼ
analyzeNetwork(layers)

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
grid

%%  ��������
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