clearvars;

% Constants
%KernelFunction = 'gaussian'; % loss = 0.6
KernelFunction = 'linear'; % loss = 0.33
Coding = 'onevsall';
K = 2;

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
fprintf('Training a muticlass model\n');
Mdl = fitcecoc(X, Y, ...
    'Coding', Coding, ...
    'Learners', t, ...
    'ClassNames',[1, 2, 3, 4], ...
    'Verbose', 2);

% Cross validate the ECOC classifier using K-fold cross validation.
fprintf('Cross validating with %d-Fold\n', K);
CVMdl = crossval(Mdl, 'kfold', K);

% Loss
loss = kfoldLoss(CVMdl, 'lossfun', 'classiferror');
fprintf('Loss %f\n', loss);