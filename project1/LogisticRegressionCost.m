function [zerooneloss, logloss, sigma, yPred] = LogisticRegressionCost(y, tX, beta)

tmp = tX * beta;

sigma = zeros(size(tmp,1),1);
sigma(tmp > 0) = 1./(1+exp(-(tmp(tmp > 0))));
sigma(tmp <= 0) = exp(tmp(tmp <= 0)) ./ (1 + exp((tmp(tmp <= 0))));

yPred = (sigma > 0.5);
wrong = sum(yPred ~= y);
zerooneloss = wrong / length(y);
%logloss = -1/length(y) * sum(y' * log(sigma) + (-y + 1)' * log(-sigma + 1));
logloss = (-1 / length(y)) * sum(y' * tX * beta - log(exp(tX * beta) + 1));

end


