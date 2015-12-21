load('train/train.mat');
load('test.mat');

% Constants
KernelFunction = 'linear';
Verbose = 2;
Coding = 'onevsall';
classes = [1 2 3 4]; % Multi-class
%classes = [-1 1]; % Binary

XTrain = [train.X_cnn train.X_hog];
XTest = [test.X_cnn test.X_hog];
yTrain = train.y;
% Binary
%yTrain(yTrain ~= 4) = -1;
%yTrain(yTrain == 4) = 1;

% Normalize data
[XTrain, mu, sigma] = zscore(XTrain);
XTest = normalize(XTest, mu, sigma);

t = templateSVM('Standardize' ,1 , ...
    'KernelFunction', KernelFunction, ...
    'KernelScale', 'auto');

Mdl = fitcecoc(XTrain, yTrain, ...
    'Coding', Coding, ...
    'Learners', t, ...
    'ClassNames',[1 2 3 4], ...
    'Verbose', Verbose);

pred = predict(Mdl, XTest, 'Verbose', Verbose);