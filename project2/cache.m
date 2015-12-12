function [] = cache(filename, X, retained)

%sigma = std(X); Fuck up the result ?
sigma = 1;
[coeff, ~, mu] = reduceDimension(normalize(X, 0, sigma), retained);

save(filename, 'coeff', 'mu', 'sigma');

end