function [coeff, score, mu, explained] = reduceDimension(M, retained)
[coeff, score, ~, ~, explained, mu] = pca(M, 'Centered', false);

% Percentage of the variance retained
sum_explained = 0;
selected = 1;

% Select columns until we reach the desired threshold
while sum_explained < retained && selected <= length(explained)
    sum_explained = sum_explained + explained(selected);
    selected = selected + 1;
end

% Return the matrice with only the selected columns
coeff = coeff(:, 1:selected);
score = score(:, 1:selected);

% M =~= score * coeff' + repmat(mean, length(M), 1);
end