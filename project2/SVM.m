clearvars;

% Constants
KernelFunction = 'linear'; % 'gaussian'
Coding = 'onevsall';

% Load features and labels of training data
load train/train.mat;

Y = train.y;
%X = csvread('reduced_xhog.csv');
X = train.X_hog;

rng(1); % For reproducibility

t = templateSVM('Standardize' ,1 , ...
    'KernelFunction', KernelFunction);

%pool = parpool; % Invoke workers
%options = statset('UseParallel', 1);

% Train an ECOC multiclass model
Mdl = fitcecoc(X, Y, ...
    'Coding', Coding, ...
    'Learners', t, ...
    'FitPosterior',1, ...
    'ClassNames',[1, 2, 3, 4], ...
    'Verbose', 2);

% Cross validate the ECOC classifier using 10-fold cross validation.
CVMdl = crossval(Mdl);
