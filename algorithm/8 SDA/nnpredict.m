function labels = nnpredict(nn, x)

%%  前向计算
    nn = nnff(nn, x, zeros(size(x, 1), nn.size(end)));

%%  得到输出结果
    [~, labels] = max(nn.a{end}, [], 2);
    
end
