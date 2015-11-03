function beta = logisticRegression( y, tX, alpha )

% Maximum number of iterations
maxIter = 10000;

% Initial beta
beta = zeros(size(tX, 2), 1);

for i = 1:maxIter
    
    tmp = tX * beta;
    
    % Compute sigma. Avoid overflow and underflow
    sigma = zeros(size(tmp,1),1);
    sigma(tmp > 0) = 1./(1+exp(-(tmp(tmp > 0))));
    sigma(tmp <= 0) = exp(tmp(tmp <= 0)) ./ (1 + exp((tmp(tmp <= 0))));
    
    % Compute gradient
    g = (tX' * (sigma - y));
   
    % Update beta
    beta = beta - alpha * g;
    
    % Convergence criteria
    if g' * g < 1e-5
        break;
    end

end


end

