function cost = LogisticRegressionCost(y, tX, beta)

cost = 0;

for n = 1:length(y)
    tmp = tX(n, :) * beta;
    cost = cost + (y(n) * tmp - log(1 + exp(tmp)));
end

end

