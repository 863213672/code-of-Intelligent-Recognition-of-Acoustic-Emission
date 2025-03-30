function [Leader_score, Leader_pos, curve] = WOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

%%  初始化参数
% 初始化位置向量和分数的领先者
Leader_pos = zeros(1, dim);
% 将其更改为 -inf 以解决最大化问题
Leader_score = inf;

%% 初始化搜索的位置
Positions = initialization(SearchAgents_no, dim, ub, lb);

%%  参数初始化
t = 0;
curve = zeros(1, Max_iter);

%%  优化算法循环
while t < Max_iter
    for i = 1: size(Positions, 1)
        
        % 返回超出搜索空间边界的搜索
        Flag4ub = Positions(i, :) > ub;
        Flag4lb = Positions(i, :) < lb;
        Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        
        % 计算每个搜索的目标函数
        fitness=fobj(Positions(i,:));
        
        % 更新领导者
        if fitness < Leader_score
            Leader_score = fitness;
            Leader_pos = Positions(i, :);
        end
        
    end
    
    % a  在方程（2.3）中线性降低从2到0
    a = 2 - t * ((2) / Max_iter); 
    
    % a2 从-1到-2的线性二分，以公式（3.12）计算 t
    a2 = -1 + t * ((-1) / Max_iter);
    
    % 更新搜索的位置
    for i = 1: size(Positions, 1)
        r1 = rand();
        r2 = rand();
        
        A = 2 * a * r1 - a;        % Eq. (2.3) in the paper
        C = 2 * r2;                % Eq. (2.4) in the paper
        b = 1;                     % parameters in Eq. (2.5)
        l = (a2 - 1) * rand + 1;   % parameters in Eq. (2.5)
        p = rand();                % parameters in Eq. (2.6)
        
        for j = 1: size(Positions, 2)
            if p < 0.5   
                if abs(A) >= 1
                    rand_leader_index = floor(SearchAgents_no * rand() + 1);
                    X_rand = Positions(rand_leader_index, :);
                    D_X_rand = abs(C * X_rand(j) - Positions(i, j)); % Eq. (2.7)
                    Positions(i, j) = X_rand(j) - A * D_X_rand;      % Eq. (2.8)
                elseif abs(A) < 1
                    D_Leader = abs(C * Leader_pos(j) - Positions(i, j)); % Eq. (2.1)
                    Positions(i, j) = Leader_pos(j) - A * D_Leader;      % Eq. (2.2)
                end
           
            elseif p >= 0.5
                % Eq. (2.5)
                distance2Leader = abs(Leader_pos(j) - Positions(i, j));
                Positions(i, j) = distance2Leader * exp(b .* l) .* cos(l .* 2 * pi) + Leader_pos(j);
            end
        end
    end

    % 保存结果
    t = t + 1;
    curve(t) = Leader_score;
end

%%  得到最优值
Leader_pos = round(Leader_pos(1: 2));