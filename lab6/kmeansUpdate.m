function [Ln,r,Mu] = kmeansUpdate(X, Mu)
% update r and Mu given X and Mu
% X is DxN data
% Mu is DxK mean vector
% r is 1xN responsibility vector, e.g. r = [1,2,1] for 2 clusters 3 data points
% Ln is 1xN minimum distance to its center for each point n

  % initialize
  K = size(Mu,2);
  [D, N] = size(X);
  r = zeros(1, N);
  Ln = zeros(1, N);

  for n = 1:N
      tmp = zeros(1, K);
      for k = 1:K
          tmp(k) = (X(:, n) - Mu(:, k))' * (X(:, n) - Mu(:, k));
      end
      [Ln(n), r(n)] = min(tmp);
      
  end
  
  for k = 1:K
      Mu(:, k) = (sum(X(:, r == k)') / length(X(:, r == k)'))';
  end
  
  % for each cluster, compute the error

  % compute r

  % compute Mu for each k

