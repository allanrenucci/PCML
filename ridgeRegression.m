function beta = ridgeRegression(y, tX, lambda)

degree = 7;
phi = [ones(size(tX, 1), 1) poly(tX, degree)];
lIm = lambda * eye(degree);
L = [zeros(1, degree + 1); zeros(degree, 1) lIm];

beta = (phi' * phi + L) \ phi' * y;

end

