function beta = leastSquaresGD( y, tX, alpha )

N = length(y);
beta = zeros(size(tX, 2), 1);
maxIter = 10000;

for i = 1:maxIter
    e = y - tX * beta;
    g = - (tX' * e) ./ N;
   
    beta = beta - alpha * g;
    
    if g' * g < 1e-5
        break;
    end    
        
end    


end

