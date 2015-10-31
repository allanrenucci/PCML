function [cost, zerooneloss, rmse] = LogisticRegressionCost(y, tX, beta)

cost = 0;
rmse = 0;
wrong = 0;

for n = 1:length(y)
    
    tmp = tX * beta;
    
    sigma = zeros(size(tmp,1),1);
    sigma(tmp > 0) = 1./(1+exp(-(tmp(tmp > 0))));
    sigma(tmp <= 0) = exp(tmp(tmp <= 0)) ./ (1 + exp((tmp(tmp <= 0))));
    
    pHat = sigma(tX(n, :) * beta); % tX(n, :) * beta
    if pHat < 0.5
        yHat = 0;
    else
        yHat = 1;
    end
    
    if yHat ~= y(n)
        wrong = wrong + 1;
    end
    
    rmse = rmse + (y(n) - pHat)^2;
    cost = cost + y(n) * log(pHat) + (1 - y(n)) * log(1 - pHat);
    
end

rmse = sqrt(rmse / length(y));
cost = cost / -length(y);
zerooneloss = wrong / length(y);

end


