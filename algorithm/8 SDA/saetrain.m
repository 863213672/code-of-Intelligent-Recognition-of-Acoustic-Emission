function sae = saetrain(sae, x, opts)

%%  训练堆叠自动编码器
    for i = 1 : numel(sae.ae)
        % 训练自动编码器
        sae.ae{i} = nntrain(sae.ae{i}, x, x, opts);
        % 前向计算
        t = nnff(sae.ae{i}, x, x);
        % 得到新的输入（输出）
        x = t.a{2};
        % 移除偏差项
        x = x(:,2:end);
    end
    
end
