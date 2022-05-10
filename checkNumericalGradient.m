
function [] = checkNumericalGradient()

x = [4; 10];
[value, grad] = simpleQuadraticFunction(x);


numgrad = computeNumericalGradient(@simpleQuadraticFunction, x);

disp([numgrad grad]);
fprintf('左边是数值梯度，右边是解析梯度\n\n');

diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); 
fprintf('数值梯度和解析梯度的范数 (需要< 1e-9)\n\n');
end


  
function [value,grad] = simpleQuadraticFunction(x)

value = x(1)^2 + 3*x(1)*x(2);

grad = zeros(2, 1);
grad(1)  = 2*x(1) + 3*x(2);
grad(2)  = 3*x(1);

end