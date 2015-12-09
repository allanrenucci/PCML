function [ ber ] = trainNeuralNetwork(XTrain, yTrain, xTest, yTest)

fprintf('Training simple neural network..\n');

addpath(genpath('DeepLearnToolbox'));

rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
opts.plot               = 1;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(XTrain) / opts.batchsize);
XTrain = XTrain(1:numSampToUse, :);
yTrain = yTrain(1:numSampToUse);

% normalize data
[normXTrain, mu, sigma] = zscore(XTrain); % train, get mu and std
Tl = reduceDimension(normXTrain);
normXTrain = normXTrain * Tl;

normXTest = normalize(xTest, mu, sigma);  % normalize test data
normXTest = normXTest * Tl;

% prepare labels for NN
LL = [1*(yTrain == 1), ...
      1*(yTrain == 2), ...
      1*(yTrain == 3), ...
      1*(yTrain == 4) ];  % first column, p(y=1)
                        % second column, p(y=2), etc

nn = nnsetup([size(normXTrain, 2) 10 4]);
nn.learningRate = 2;                        
[nn, L] = nntrain(nn, normXTrain, LL, opts);

% to get the scores we need to do nnff (feed-forward)
%  see for example nnpredict().
% (This is a weird thing of this toolbox)
nn.testing = 1;
nn = nnff(nn, normXTest, zeros(size(normXTest, 1), nn.size(end)));
nn.testing = 0;


% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~, classVote] = max(nnPred, [], 2);

% get overall error [NOTE!! this is not the BER, you have to write the code
%                    to compute the BER!]
predErr = sum(classVote ~= yTest) / length(yTest);
ber = BER(4, yTest, classVote);

fprintf('\nTesting error: %.2f%%, ber=%f\n\n', predErr * 100, ber);

end

