%%  清空环境变量
clc
clear all
close all
warning off

%%  计算标准差，并筛选数据
aresult = xlsread('数据集.xlsx');
for i=1:size(aresult,2)
    biaozhuncha(i,1)=std(aresult(:,i));
    junzhi(i,1)=mean(aresult(:,i));
end
for ai=1:size(aresult,1)
    for bi=1:size(aresult,2)
    if abs(aresult(ai, bi)-junzhi(bi,1))>2*biaozhuncha(bi,1)
        A(ai,bi)=0;
    else
        A(ai,bi)=1;
    end
    end 
end
%% 
for ci=1:size(A,1)
    B(ci,1)=sum(A(ci,:));
    if B(ci,1)<size(A,2)
        C(ci,1)=0;
    else
        C(ci,1)=1;
    end
end

j=0;
for di=1:size(C,1)
    if C(di,1)==1
        j=j+1;
        result(j,:)=aresult(di,:);
        D(j,1)=di;
    end
end

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
Z(:,4)=D;
%% 
u=0;
uu=0;
result=result';
for ei=1:size(Z,1)
    if Z(ei,1)==1
        u=u+1;
        ZA(u,:)=aresult(Z(ei,4),:);
    end
        
        if Z(ei,1)==2
        uu=uu+1;
        ZAA(uu,:)=aresult(Z(ei,4),:);
        end
end
%% 
Q=length(find(T_sim1(1,1:24367)==2));
QQ=length(find(T_sim1(1,24368:49819)==1));
Z(1:24367,5)=2;
Z(24368:49819,5)=1;
%% 
j=0;
J=0;
k=0;
K=0;
for qi=1:24367
    if Z(qi,1)==2 && Z(qi,5)==2
        j=j+1;
        ZZhao(j,:)=Z(qi,:);
    end
    
    if Z(qi,1)==1 && Z(qi,5)==2
        J=J+1;
        ZQian(J,:)=Z(qi,:);
    end
end


for qi=24368:49819
    if Z(qi,1)==1 && Z(qi,5)==1
        k=k+1;
        ZSun(k,:)=Z(qi,:);
    end
    
    if Z(qi,1)==2 && Z(qi,5)==1
        K=K+1;
        ZLi(K,:)=Z(qi,:);
    end
end




