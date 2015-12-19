function err = neuralNetwork(XTrain, yTrain, XTest, yTest, coeff)

fprintf('Training simple neural network..\n');

addpath(genpath('DeepLearnToolbox'));

rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
% opts.plot               = 1;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(XTrain) / opts.batchsize);
XTrain = XTrain(1:numSampToUse, :);
yTrain = yTrain(1:numSampToUse);

% Apply dimensionality reduction 
XTrain = XTrain * coeff;
XTest = XTest * coeff;


% prepare labels for NN
LL = [1*(yTrain == 1), ...
      1*(yTrain == 2), ...
      1*(yTrain == 3), ...
      1*(yTrain == 4) ];  % first column, p(y=1)
                        % second column, p(y=2), etc

nn = nnsetup([size(XTrain, 2) 10 4]);
nn.learningRate = 2;                        
[nn, L] = nntrain(nn, XTrain, LL, opts);

%%%%
nn.testing = 1;
nn = nnff(nn, XTrain, zeros(size(XTrain, 1), nn.size(end)));
nn.testing = 0;
nnPredTrain = nn.a{end};
[~, classVoteTrain] = max(nnPredTrain, [], 2);
%%%%


% to get the scores we need to do nnff (feed-forward)
%  see for example nnpredict().
% (This is a weird thing of this toolbox)
nn.testing = 1;
nn = nnff(nn, XTest, zeros(size(XTest, 1), nn.size(end)));
nn.testing = 0;

% predict on the test set
nnPredTest = nn.a{end};

% get the most likely class
[~, classVoteTest] = max(nnPredTest, [], 2);

% get overall error [NOTE!! this is not the BER, you have to write the code
%                    to compute the BER!]
%predErrTrain = sum(

predErrTrain = sum(classVoteTrain ~= yTrain) / length(yTrain);
berTrain = BER(4, yTrain, classVoteTrain);

predErrTest = sum(classVoteTest ~= yTest) / length(yTest);
berTest = BER(4, yTest, classVoteTest);

fprintf('\nTesting error: %.2f%%, ber=%f\n\n', predErrTest * 100, berTest);
err = [predErrTrain, berTrain, predErrTest, berTest];
end
