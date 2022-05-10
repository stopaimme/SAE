function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

%将长向量转换成每一层的权值矩阵和偏置向量值
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% 初始化损失和梯度 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));


Jcost = 0;%直接误差
Jweight = 0;%权值惩罚
Jsparse = 0;%稀疏性惩罚
[n m] = size(data);%m为样本的个数，n为样本的特征数


%前向算法计算各神经网络节点的线性组合值和active值
z2 = W1*data+repmat(b1,1,m);%注意这里一定要将b1向量复制扩展成m列的矩阵
a2 = sigmoid(z2);
z3 = W2*a2+repmat(b2,1,m);
a3 = sigmoid(z3);

% 计算预测产生的误差
Jcost = (0.5/m)*sum(sum((a3-data).^2));

%计算权值惩罚项
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));

%计算稀释性规则项
rho = (1/m).*sum(a2,2);%求出第一个隐含层的平均值向量
Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
        (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));

%损失函数的总表达式
cost = Jcost+lambda*Jweight+beta*Jsparse;

%反向算法求出每个节点的误差值
d3 = -(data-a3).*sigmoidInv(z3);
sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));%因为加入了稀疏规则项，所以
                                                             %计算偏导时需要引入该项
d2 = (W2'*d3+repmat(sterm,1,m)).*sigmoidInv(z2); 

%计算W1grad 
W1grad = W1grad+d2*data';
W1grad = (1/m)*W1grad+lambda*W1;

%计算W2grad  
W2grad = W2grad+d3*a2';
W2grad = (1/m).*W2grad+lambda*W2;

%计算b1grad 
b1grad = b1grad+sum(d2,2);
b1grad = (1/m)*b1grad;%注意b的偏导是一个向量，所以这里应该把每一行的值累加起来

%计算b2grad 
b2grad = b2grad+sum(d3,2);
b2grad = (1/m)*b2grad;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%sigmoid函数
function sigm = sigmoid(x)

    sigm = 1 ./ (1 + exp(-x));
end

%sigmoid函数的逆向求导函数
function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end