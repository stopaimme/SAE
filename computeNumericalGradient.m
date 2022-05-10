function numgrad = computeNumericalGradient(J, theta)
numgrad = zeros(size(theta));
epsilon = 1e-4;
n = size(theta,1);
E = eye(n);
for i = 1:n
    delta = E(:,i)*epsilon;
    numgrad(i) = (J(theta+delta)-J(theta-delta))/(epsilon*2.0);
end

end