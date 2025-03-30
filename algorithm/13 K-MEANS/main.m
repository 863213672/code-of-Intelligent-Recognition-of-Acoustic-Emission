%%  清空环境变量
clc
clear all
close all
warning off
%%  导入数据
result = xlsread('数据集.xlsx');

%%  参数设置
M  = size(result, 1);          % 样本数目

%%  输入特征
result = result';

%%  数据归一化
[p_train, ps_input] = mapminmax(result, 0, 1);

%%  矩阵转置
p_train = p_train';

%%  聚类算法
num_class = 2;              % 聚类类别数目
[T_sim1, C, sumD] = kmeans(p_train, num_class); 


%%  TSNE -- 降维
pc_train = tsne(p_train);

%%  绘制结果图
figure
gscatter(pc_train(:, 1), pc_train(:, 2), T_sim1)
xlabel('降维后第一维度')
ylabel('降维后第二维度')
string = {'聚类可视化'};
title(string)
grid on
%% 
T_sim1=T_sim1';
Z(:,1)=T_sim1;
Z(:,2:3)=pc_train;
Z(:,4)=6;
%% 
Q=length(find(T_sim1(1,1:44000)==1));
QQ=length(find(T_sim1(1,44001:88000)==2));
Z(1:44000,5)=2;
Z(44001:88000,5)=1;
%% 
j=0;
J=0;
k=0;
K=0;
for qi=1:88000
    if Z(qi,1)==2 && Z(qi,5)==2
        j=j+1;
        ZZhao(j,:)=Z(qi,:);
    end
    
    if Z(qi,1)==1 && Z(qi,5)==2
        J=J+1;
        ZQian(J,:)=Z(qi,:);
    end
    
    if Z(qi,1)==1 && Z(qi,5)==1
        k=k+1;
        ZSun(k,:)=Z(qi,:);
    end
    
    if Z(qi,1)==2 && Z(qi,5)==1
        K=K+1;
        ZLi(K,:)=Z(qi,:);
    end
end



