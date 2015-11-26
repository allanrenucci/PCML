%% Clear workspace

% INSERT CODE WHERE INDICATED
% Written by Benoit Seguin & Victor Kristof, EPFL for PCML 2015
% All rights reserved

clear;
close all;

%% Exercise 1.1

load('two_spirals.mat')
X = two_spirals(:, 1:2);
y = two_spirals(:, 3);
N = length(y);

% Visualization
figure(1);
plot(X(y==1, 1), X(y==1, 2), 'ob'); hold on;
plot(X(y==-1, 1), X(y==-1, 2), 'or');
xlim([min(X(:, 1)) max(X(:, 1))]);
ylim([min(X(:, 2)) max(X(:, 2))]);

% Normalize features (store the mean and variance)
mean_X = mean(X);
X_norm = X - ones(N, 1) * mean_X;
std_X = std(X_norm);
X_norm = X_norm*diag(1 ./ std_X);
tX = [ones(N, 1) X_norm];

% ENTER CODE FOR PENALIZED LOGISTIC REGRESSION
betas = randn(1, 3);

% Create a 2D meshgrid of values of heights and weights
h = min(X(:,1)):.01:max(X(:,1));
w = min(X(:,2)):1:max(X(:,2));
[hx, wx] = meshgrid(h,w);
% Predict for each pair, i.e. create tX for each [hx,wx]
% and then predict the value. After that you should
% reshape `pred` so that you can use `contourf`.
% For this you need to understand how `meshgrid` works.

N_test = length(hx(:));
X_pred = [hx(:), wx(:)];
X_pred_norm = X_pred - ones(N_test,1) * mean_X;
X_pred_norm = X_pred_norm * diag(1./std_X);

% Form (y,tX) to get regression data in matrix form
% tX_pred = [ones(N_test,1),X_pred_norm];
tX_pred = [ones(N_test, 1) X_pred_norm];

% IMPLEMENT PREDICTION FOR EACH TEST POINT
pred = 0;

pred = reshape(pred, size(hx));

% Plot the decision surface
figure(2)
contourf(hx, wx, pred, 1); hold on;
%colormap(jet(4))
% Plot indiviual data points
males = (y==1);
females = (y==-1);
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(X(males,1), X(males,2), 'xr', 'color', myRed, 'linewidth', 2, ...
'markerfacecolor', myRed); hold on;
plot(X(females,1), X(females,2),'or','color', ...
myBlue,'linewidth', 2, 'markerfacecolor', myBlue); hold on;
xlabel('height');
ylabel('weight');
xlim([min(h) max(h)]);
ylim([min(w) max(w)]);
grid on;

%% Exercise 1.2

load('height_weight_gender.mat');
height = height * 0.025;
weight = weight * 0.454;
y = double(gender);
y(y==0) = -1;
X = [height(:) weight(:)];
N = length(y);

% Randomly permute data
idx = randperm(N);
y = y(idx);
X = X(idx,:);
% Subsample
y = y(1:200);
X = X(1:200, :);
N = length(y);

% Normalize features (store the mean and variance)
mean_X = mean(X);
X_norm = X - ones(N, 1) * mean_X;
std_X = std(X_norm);
X_norm = X_norm*diag(1 ./ std_X);
tX = X_norm;

% COMPLETE linear_kernel.m WITH CODE FOR THE LINEAR KERNEL
K = linear_kernel(tX, tX);

% Compute parameters
C = 0.1;
[alphas, beta0] = SMO(K, y, C);

% Visualize SVM classification and margins
% Create a 2D meshgrid of values of heights and weights
h = min(X(:,1)):.01:max(X(:,1));
w = min(X(:,2)):1:max(X(:,2));
[hx, wx] = meshgrid(h,w);
% Predict for each pair, i.e. create tX for each [hx,wx]
% and then predict the value. After that you should
% reshape `pred` so that you can use `contourf`.
% For this you need to understand how `meshgrid` works.

N_test = length(hx(:));
X_pred = [hx(:), wx(:)];
X_pred_norm = X_pred - ones(N_test,1) * mean_X;
X_pred_norm = X_pred_norm * diag(1./std_X);

% Form (y,tX) to get regression data in matrix form
tX_pred = X_pred_norm;

% IMPLEMENT PREDICTION FOR EACH TEST POINT
pred = ones(size(hx, 1) * size(hx, 2), 1);

pred = reshape(pred, size(hx));

% Plot the decision surface
contourf(hx, wx, pred, [1, 0, -1, -4]); hold on;
colormap(jet(5))
% Plot indiviual data points
males = (y==1);
females = (y==-1);
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(X(males,1), X(males,2), 'xr', 'color', myRed, 'linewidth', 2, ...
'markerfacecolor', myRed); hold on;
plot(X(females,1), X(females,2),'or','color', ...
myBlue,'linewidth', 2, 'markerfacecolor', myBlue); hold on;
% Plot support vectors
plot(X(alphas>0, 1), X(alphas>0, 2),'o','linewidth', 2, 'color', 'black');
xlabel('height');
ylabel('weight');
xlim([min(h) max(h)]);
ylim([min(w) max(w)]);
grid on;

%% Exercise 1.3

load('two_spirals.mat')
X = two_spirals(:, 1:2);
y = two_spirals(:, 3);
N = length(y);

% Randomly permute data
idx = randperm(N);
y = y(idx);
X = X(idx,:);
% Subsample
y = y(1:2000);
X = X(1:2000, :);
N = length(y);

% Normalize features (store the mean and variance)
mean_X = mean(X);
X_norm = X - ones(N, 1) * mean_X;
std_X = std(X_norm);
X_norm = X_norm*diag(1 ./ std_X);
tX = X_norm;

% COMPLETE rbf_kernel.m WITH CODE FOR THE RBF KERNEL
gamma = 1;
K = rbf_kernel(tX, tX, gamma);

% Compute parameters
C = 1;
[alphas, beta0] = SMO(K, y, C);

% Visualize SVM classification and margins
% Create a 2D meshgrid of values of heights and weights
h = min(X(:,1)):.01:max(X(:,1));
w = min(X(:,2)):1:max(X(:,2));
[hx, wx] = meshgrid(h,w);
% Predict for each pair, i.e. create tX for each [hx,wx]
% and then predict the value. After that you should
% reshape `pred` so that you can use `contourf`.
% For this you need to understand how `meshgrid` works.

N_test = length(hx(:));
X_pred = [hx(:), wx(:)];
X_pred_norm = X_pred - ones(N_test,1) * mean_X;
X_pred_norm = X_pred_norm * diag(1./std_X);

% Form (y,tX) to get regression data in matrix form
% tX_pred = [ones(N_test,1),X_pred_norm];
tX_pred = X_pred_norm;
kernel_pred = rbf_kernel(tX_pred, tX, gamma);

% IMPLEMENT PREDICTION FOR EACH TEST POINT
pred = ones(size(hx, 1) * size(hx, 2), 1);

pred = reshape(pred, size(hx));

% Plot the decision surface
contourf(hx, wx, pred, [1, 0, -1, -2]); hold on;
%colormap(jet(6))
% Plot indiviual data points
males = (y==1);
females = (y==-1);
myBlue = [0.06 0.06 1];
myRed = [1 0.06 0.06];
plot(X(males,1), X(males,2), 'xr', 'color', myRed, 'linewidth', 2, ...
'markerfacecolor', myRed); hold on;
plot(X(females,1), X(females,2),'or','color', ...
myBlue,'linewidth', 2, 'markerfacecolor', myBlue); hold on;
% Plot support vectors
plot(X(alphas>0, 1), X(alphas>0, 2),'o','linewidth', 2, 'color', 'black');
xlim([min(h) max(h)]);
ylim([min(w) max(w)]);
grid on;
