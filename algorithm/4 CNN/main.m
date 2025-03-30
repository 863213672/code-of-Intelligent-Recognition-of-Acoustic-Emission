%%  ��ջ�������
clc                     % ���������
clear all                  % ��ձ���
close all               % �رտ�����ͼ��
warning off             % �رձ�����Ϣ
%%  ��������

res = xlsread('���ݼ�.xlsx');
%%  ����ѵ�����Ͳ��Լ�
qq=size(res,2);
aqq=88000;
P_train = res(1: aqq, 1: qq-1)';
T_train = res(1: aqq, qq)';
M = size(P_train, 2);

P_test = res(aqq+1: end, 1: qq-1)';
T_test = res(aqq+1: end, qq)';
N = size(P_test, 2);
%%  ���ݹ�һ��
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';
%%  ����ƽ��
%   ������ƽ�̳�1ά����ֻ��һ�ִ���ʽ
%   Ҳ����ƽ�̳�2ά���ݣ��Լ�3ά���ݣ���Ҫ�޸Ķ�Ӧģ�ͽṹ
%   ����Ӧ��ʼ�պ���������ݽṹ����һ��
p_train =  double(reshape(P_train, qq-1, 1, 1, M));
p_test  =  double(reshape(P_test , qq-1, 1, 1, N));

%%  ��������ṹ
layers = [
 imageInputLayer([qq-1, 1, 1])                                % �����
 
 convolution2dLayer([2, 1], 16, 'Padding', 'same')          % ����˴�СΪ 2*1 ����16�����
 batchNormalizationLayer                                    % ����һ����
 reluLayer                                                  % relu �����
 
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % ���ػ��� ��СΪ 2*1 ����Ϊ [2, 1]

 convolution2dLayer([2, 1], 32, 'Padding', 'same')          % ����˴�СΪ 2*1 ����32�����
 batchNormalizationLayer                                    % ����һ����
 reluLayer                                                  % relu �����

 fullyConnectedLayer(2)                                     % ȫ���Ӳ㣨������� 
 softmaxLayer                                               % ��ʧ������
 classificationLayer];                                      % �����

%%  ��������
options = trainingOptions('adam', ...      % Adam �ݶ��½��㷨
    'MaxEpochs', 500, ...                  % ���ѵ������ 500
    'InitialLearnRate', 1e-3, ...          % ��ʼѧϰ��Ϊ 0.001
    'L2Regularization', 1e-4, ...          % L2���򻯲���
    'LearnRateSchedule', 'piecewise', ...  % ѧϰ���½�
    'LearnRateDropFactor', 0.1, ...        % ѧϰ���½����� 0.1
    'LearnRateDropPeriod', 400, ...        % ����450��ѵ���� ѧϰ��Ϊ 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % ÿ��ѵ���������ݼ�
    'ValidationPatience', Inf, ...         % �ر���֤
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
c=T_sim2(1,1:1000);
d=T_sim2(1,1001:2000);
e=length(find(c==1));
f=length(find(d==2));
%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  �����������ͼ
analyzeNetwork(layers)

%%  ��������
[T_train, index_1] = sort(T_train);
% [T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
% T_sim2 = T_sim2(index_2);

%%  ��ͼ
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

%%  ��������
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
