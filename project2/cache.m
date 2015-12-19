function [] = cache(filename, X, retained)

[coeff, score, mu] = reduceDimension(X, retained);

save(filename, 'X', 'coeff', 'score', 'mu');

end