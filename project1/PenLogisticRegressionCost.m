function cost = PenLogisticRegressionCost(y, tX, beta, lambda)

cost = 0;

for n = 1:length(y)
    tmp = tX(n, :) * beta;
    cost = cost + (y(n) * tmp - log(1 + exp(tmp))) + lambda * (beta' * beta);
end

end

