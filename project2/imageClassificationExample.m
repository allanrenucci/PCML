clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %
% IMPORTANT:
%    Make sure you downloaded the file train.tar.gz provided to you
%    and uncompressed it in the same folder as this file resides.

% Load features and labels of training data
load train/train.mat;

%% --browse through the images and look at labels
for i=1:10
    clf();

    % load img
    img = imread( sprintf('train/imgs/train%05d.jpg', i) );

    % show img
    imshow(img);

    pred = train.y(i);
    title(sprintf('Label %s', labelToString(pred)));

    %pause;  % wait for key,?? 
end

%% -- Example: split half and half into train/test, use HOG features
fprintf('Splitting into train/test..\n');

Tr = [];
Te = [];

% NOTE: you should do this randomly! and k-fold!
K = 2;
c = cvpartition(train.y, 'KFold', K);

fun = @(xTrain, yTrain, xTest, yTest)(trainNeuralNetwork(xTrain, yTrain, xTest, yTest));
errors = crossval(fun, train.X_hog, train.y, 'partition', c);
