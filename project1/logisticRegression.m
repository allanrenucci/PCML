function beta = logisticRegression( y, tX, alpha )

N = length(y);
beta = zeros(size(tX, 2), 1);
maxIter = 10000;

for i = 1:maxIter
    e = y - tX * beta;
    sigma = 1./(1+exp(-(tX * beta)));
    g = tX' * (sigma - y);
   
    beta = beta - alpha * g;
    
    if g' * g < 1e-5
        disp('converged');
        break;
    end
    
        
end

end
