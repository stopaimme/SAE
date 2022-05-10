%% STEP 0:设置参数
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

visibleSize = 8*8;   % 输入大小
hiddenSize = 25;     % 隐藏层大小
sparsityParam = 0.01;   % 稀疏参数
                    
             %  in the lecture notes). 
lambda = 0.0001;     % 权重衰减 
beta = 3;            % 系数惩罚项     

%% STEP 1:获取数据集
patches = sampleIMAGES;
display_network(patches(:,randi(size(patches,2),204,1)),8)
theta = initializeParameters(hiddenSize, visibleSize);

%% STEP 2: 获得梯度和损失
[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, patches);

%% STEP 3: 梯度校验

checkNumericalGradient();

numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                  patches), theta);

diff = norm(numgrad-grad)/norm(numgrad+grad); %数值梯度-解析梯度
disp(diff); 

       
%% STEP 4: 训练
theta = initializeParameters(hiddenSize, visibleSize);%初始化参数

%  Use minFunc to minimize the function
addpath C:\Users\ss\Downloads\liji597760593-6633087-minFunc_2012\minFunc_2012\minFunc\
options.Method = 'lbfgs'; 
options.maxIter = 400;      % 最大迭代次数
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

%% STEP 5: 可视化隐藏层

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
figure;
display_network(W1', 12); 


print -djpeg weights.jpg   % save the visualization to a file
