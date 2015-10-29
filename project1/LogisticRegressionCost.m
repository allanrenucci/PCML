function [cost, zerooneloss] = LogisticRegressionCost(y, tX, beta)

cost = 0;
correct = 0;

for n = 1:length(y)
    tmp = tX(n, :) * beta;
    if tmp < 0.5
        tmp = 0;
    else
        tmp = 1;
    end
    %fprintf('Predicted = %d, actual = %d\n', tmp, y(n));
    if tmp == y(n)
        correct = correct + 1;
    end
    cost = cost + (y(n) * tmp - log(1 + exp(tmp)));
    zerooneloss = correct / length(y);
end

%fprintf('%d / %d -> %f%%\n', correct, length(y), correct / length(y) * 100);

end


