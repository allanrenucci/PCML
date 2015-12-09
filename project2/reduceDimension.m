function coeff = reduceDimension(M)
[coeff, ~, ~, ~, explained] = pca(M);

% Percentage of the variance that we want to keep
threshold = 95;
sum_explained = 0;
selected = 1;

% Select columns until we reach the desired threshold
while sum_explained < threshold && selected <= length(explained)
    sum_explained = sum_explained + explained(selected);
    selected = selected + 1;
end

% Return the matrice with only the selected columns
coeff = coeff(:, 1:selected);

% M =~= score * coeff' + repmat(mean, length(M), 1);
end