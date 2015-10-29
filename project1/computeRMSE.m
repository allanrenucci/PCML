function rmse = computeRMSE(y, tX, beta)
%Compute RMSE

N = length(y);
e = y - tX * beta;
rmse = sqrt(e' * e / N);

end

