function beta = leastSquaresGD( y, tX, alpha )

% Maximum number of iterations
maxIter = 10000;

N = length(y);

% Initial beta
beta = zeros(size(tX, 2), 1);

for i = 1:maxIter
	% Compute error
    e = y - tX * beta;

    % Compute gradient
    g = - (tX' * e) ./ N;
   
    % Update beta
    beta = beta - alpha * g;
    
    % Convergence criteria
    if g' * g < 1e-5
        break;
    end    
        
end    


end

