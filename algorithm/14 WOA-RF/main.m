%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ���·��
addpath('ClassRF\')

%%  ��ȡ����
res = xlsread('���ݼ�.xlsx');

%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.7;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�
flag_conusion = 1;                        % ��־λΪ1���򿪻�������Ҫ��2018�汾�����ϣ�
f_ = size(res, 2) - 1;                    % ����������������Ŀ

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
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
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  ����ת��
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  ��������
fun = @getObjValue;                 % Ŀ�꺯��
dim = 2;                            % �Ż���������
lb  = [100, 01];                    % �Ż�����Ŀ�����ޣ���������Ŀ��������ȣ�
ub  = [500, f_];                    % �Ż�����Ŀ�����ޣ���������Ŀ��������ȣ�
pop = 10;                           % ��������
Max_iteration = 50;                 % ����������

%%  �Ż��㷨
[Best_score, Best_pos, curve] = WOA(pop, Max_iteration, lb, ub, dim, fun);

%%  ��ȡ���Ų���
ntree = Best_pos(1, 1);          % number of trees
mtry  = Best_pos(1, 2);          % default is floor(sqrt(size(X,2))

%%  ����ģ��
% extra_options.sampsize = 80; % Size of sample to draw. 
% extra_options.nodesize = 5;  % Minimum size of terminal nodes.
% model = regRF_train(p_train, t_train, ntree, mtry, extra_options);
model = classRF_train(p_train, t_train, ntree, mtry);

%%  ������Ҫ��
importance = model.importance';

%%  �������
[T_sim1, Vote1] = classRF_predict(p_train, model);
[T_sim2, Vote2] = classRF_predict(p_test , model);

%%  ��������
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  ��Ӧ������
figure;
plot(1: length(curve), curve, 'LineWidth', 1.5);
title('��Ӧ������', 'FontSize', 13);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��ֵ', 'FontSize', 10);
xlim([1, length(curve)])
grid

%%  ����������Ҫ��
figure
bar(model.importance)
legend('��Ҫ��')
xlabel('����')
ylabel('��Ҫ��')

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