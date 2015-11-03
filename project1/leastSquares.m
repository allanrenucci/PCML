function beta = leastSquares( y, tX )

% Compute beta
beta = (tX' * tX) \ tX' * y;

end

