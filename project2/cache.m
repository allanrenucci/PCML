function [] = cache(filename, X, retained)

[coeff, score, mu, explained] = reduceDimension(X, retained);

save(filename, 'X', 'coeff', 'score', 'mu', 'explained');

end