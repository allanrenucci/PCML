function results = runOnTrees(trees, X, waitBar)
    numTrees = size(trees, 1);
    predictions = zeros(size(X, 1), size(trees, 1));
    waitbar(0.5, waitBar, 'Running predictions...');
    
    % Get the prediction of each tree
    for i = 1:numTrees
        waitbar(0.5 + i / numTrees, waitBar, strcat('Running predictions on tree ', num2str(i), '...'));
        predictions(:, i) = predict(trees{i}, X);
    end
    
    % Return the majority
    results = mode(predictions, 2);
end