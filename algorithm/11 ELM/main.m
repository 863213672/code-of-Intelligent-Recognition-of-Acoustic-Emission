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
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = T_train;
t_test  = T_test ;
%%  ����ģ��
num_hiddens = 50;        % ���ز�ڵ����
activate_model = 'sig';  % �����
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 1);

%%  �������
T_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);
T_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);

%%  ��������
[T_train, index_1] = sort(T_train);
% [T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
% T_sim2 = T_sim2(index_2);

%%  ��������
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

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