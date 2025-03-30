%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ���·��
addpath("RF_Toolbox\")

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

%%  ת������Ӧģ��
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  ������ʼ��
c1      =  4.495;            % ѧϰ����
c2      =  4.494;            % ѧϰ����
maxgen  =    50;             % ��Ⱥ���´���  
sizepop =     5;             % ��Ⱥ��ģ
Vmax    = [ 100.0,  2.0];    % ����ٶ�(������Ŀ���������)
Vmin    = [-100.0, -2.0];    % ��С�ٶ�(������Ŀ���������)
popmax  = [  800, f_] ;      % ���߽�(������Ŀ��������ȣ������Ȳ��ܳ���������)
popmin  = [  200,  1];       % ��С�߽�(������Ŀ��������ȣ���С���Ϊ1)

%%  ��Ⱥ��ʼ��
for i = 1 : sizepop
    pop(i, :) = rand(1, length(popmax)) .* (popmax - popmin) + popmin;
    V(i, :) = rand(1, length(Vmax)) .* Vmax + 1;
    fitness(i) = fun(pop(i, :), p_train, t_train);
end

%%  ���弫ֵ��Ⱥ�弫ֵ
[fitnesszbest, bestindex] = min(fitness);
zbest = pop(bestindex, :);     % ȫ�����
gbest = pop;                   % �������
fitnessgbest = fitness;        % ���������Ӧ��ֵ
BestFit = fitnesszbest;        % ȫ�������Ӧ��ֵ

%%  ����Ѱ��
for i = 1 : maxgen
    for j = 1 : sizepop

        % �ٶȸ���
        V(j, :) = V(j, :) + c1 * rand * (gbest(j, :) - pop(j, :)) + c2 * rand * (zbest - pop(j, :));
        
        if (sum(V(j, :) > Vmax) > 0 || sum(V(j, :) < Vmin) > 0)
            [~, vmax_index] = find(V(j, :) > Vmax);
            [~, vmin_index] = find(V(j, :) < Vmin);
            V(j, vmax_index) = Vmax(vmax_index);
            V(j, vmin_index) = Vmin(vmin_index);
        end
        
        % ��Ⱥ����
        pop(j, :) = pop(j, :) + 0.2 * V(j, :);

        if (sum(pop(j, :) > popmax) > 0 || sum(pop(j, :) < popmin) > 0)
            [~, pmax_index] = find(pop(j, :) > popmax);
            [~, pmin_index] = find(pop(j, :) < popmin);
            pop(j, pmax_index) = popmax(pmax_index);
            pop(j, pmin_index) = popmin(pmin_index);
        end
        
        % ����Ӧ����
        if rand > 0.95
            pop(j, :) = rand(1, length(popmax)) .* (popmax - popmin) + 1;
        end
        
        % ��Ӧ��ֵ
        fitness(j) = fun(pop(j, :), p_train, t_train);
    end
    
    for j = 1 : sizepop
        
        % �������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j, :) = pop(j, :);
            fitnessgbest(j) = fitness(j);
        end

        % Ⱥ�����Ÿ��� 
        if fitness(j) < fitnesszbest
            zbest = pop(j, :);
            fitnesszbest = fitness(j);
        end

    end

    BestFit = [BestFit, fitnesszbest];    
end

%%  ��ȡ���Ų���
n_trees = round(zbest(1));
n_layer = round(zbest(2));

%%  ����ģ��
model = classRF_train(p_train, t_train, n_trees, n_layer);
importance = model.importance; % ��������Ҫ��

%%  �������
[T_sim1, Vote1] = classRF_predict(p_train, model);
[T_sim2, Vote2] = classRF_predict(p_test , model);

%%  ��������
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  ��Ӧ������
figure;
plot(1: length(BestFit), BestFit, 'LineWidth', 1.5);
title('��Ӧ������', 'FontSize', 13);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��ֵ', 'FontSize', 10);
xlim([1, length(BestFit)])
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