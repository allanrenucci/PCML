% Recommendation system
% Written by Emtiyaz, EPFL
% Modified on Nov 17, 2015
clear all

% load data
load('movielens100k.mat');
X = ratings;
[D,N] = size(X);

% Number of users who rated a movie 
% and number of movies rated per user 
num_movies_per_user = sum(X~=0, 1);
num_users_per_movie = sum(X~=0, 2);

figure(1);
subplot(121); 
plot(sort(num_movies_per_user, 'descend'), 'linewidth', 2);
ylabel('Number of ratings (sorted)');
xlabel('Users');
grid on;
subplot(122); 
plot(sort(num_users_per_movie, 'descend'), 'linewidth', 2);
ylabel('Number of ratings (sorted)');
xlabel('Movies');
grid on;

%%%%%%%%%%%%%
% Make (sparse) train set X and test set Xtest
% by punching holes in matrix X.
% It is important to understand this to generalize well.
%%%%%%%%%%%%%
% First, set the seed
setSeed(999); 
% The maximum number of movies held out per user
numD = 5; 
% Index of users with more than 10 ratings
idx_user = find(num_movies_per_user > 10); 
% Index of movies with more than 10 ratings
idx_movies = find(num_users_per_movie > 10);
% For all users with more than 10 ratings
dd = []; nn = []; xx = [];
for n = idx_user
    % Randomly select a subset of ratings
    On = find(X(:, n) ~= 0); 
    idx = unidrnd(length(On), numD, 1);
    d = On(idx);
    % But only choose movies with more than 10 ratings
    d = intersect(d, idx_movies); 
    % And then add them to test set
    dd = [dd; d];
    nn = [nn; n * ones(length(d), 1)];
    xx = [xx; X(d, n)];
end
% Create the test matrix
Xtest = sparse(dd, nn, xx, D, N);
% Create the train matrix
X(sub2ind([D N], dd, nn)) = 0;
%%%%%%%%%%%%%

% Visualize the test and training sets
figure(2)
subplot(121);
spy(X);
title('Training data');
xlabel('Users');
ylabel('Movies');
subplot(122);
spy(Xtest);
title('Test data');
xlabel('Users');
ylabel('Movies');

% Baseline 1 : global mean
obs = find(X ~= 0);
global_mean = mean(X(obs));
obs = find(Xtest ~= 0);
rmse_gm = sqrt(mean((Xtest(obs) - global_mean) .^ 2));
fprintf('RMSE for global mean: %.4f\n', full(rmse_gm));

% Baseline 2 : user's mean
% INSERT CODE
for n = 1:size(X, 2)
    X_n = X(:, n);
    user_mean(n) = mean(X_n(X_n ~= 0));
end

user_mean = repmat(user_mean, size(X, 1), 1);
obs = Xtest ~= 0;
rmse_um = sqrt(mean((Xtest(obs) - user_mean(obs)) .^ 2));

fprintf('RMSE for user mean: %.4f\n', full(rmse_um));

% Baseline 3 : movies's mean
% INSERT CODE
fprintf('RMSE for movie mean: %.4f\n', NaN);

%%%%%%%%%%%
% ALS
%%%%%%%%%%%
M = 2;
lambda_z = 1;
lambda_w = 1;
max_iters = 5;

% Random start
W = randn(D, M);
Z = randn(N, M);

% Iterate
for iter = 1:max_iters;
  % Update Z given W
  % INSERT CODE

  % Update W given Z
  % INSERT CODE

  % predict
  Xhat = W * Z';
  obs = find(Xtest~=0);
  rmse(iter) = sqrt(mean((Xtest(obs) - Xhat(obs)).^2));
  fprintf('ALS iteration (%d) %.4f\n', iter, full(rmse(iter)));
end

