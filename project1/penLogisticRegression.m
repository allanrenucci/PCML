function beta = penLogisticRegression( y, tX, alpha, lambda )

N = length(y);
beta = zeros(size(tX, 2), 1);
maxIter = 100000;

for i = 1:maxIter
    
    tmp = tX * beta;
    
    sigma = zeros(size(tmp,1),1);
    sigma(tmp > 0) = 1./(1+exp(-(tmp(tmp > 0))));
    sigma(tmp <= 0) = exp(tmp(tmp <= 0)) ./ (1 + exp((tmp(tmp <= 0))));
    
    g = (tX' * (sigma - y)) + 2 * lambda * beta;
   
    beta = beta - alpha * g;
    
    if g' * g < 1e-5
        disp('conv');
        break;
    end
    
    if i == maxIter
        disp('div');
    end
end

end

