function [opts] = initTrainParams()
    
    % add pathes
    addpath(genpath('../../SiamLT/'));
    startup();    
    % set default parameters
    p = initParams();
    
    opts.features = p.features;
    opts.feat_global = p.feat_global;
    
    opts.net.type = 'alexnet';
    opts.net.conf = struct(); % Options depend on type of net.
    opts.pretrain = false; % Location of model file set in env_paths.
    
    opts.init.scale = 1;
    opts.init.weightInitMethod = 'xavierimproved';
    opts.init.initBias = 0.1;
    
    opts.expDir = 'data'; % where to save the trained net
    opts.numFetchThreads = 12; % used by vl_imreadjpg when reading dataset
    opts.validation = 0.1; % fraction of imbd reserved to validation
    
    opts.exemplarSize = p.exemplarSize; % exemplar (z) in the paper
    opts.instanceSize = p.instanceSize; % search region (x) in the paper
    
    opts.loss.type = 'simple';
    opts.loss.rPos = 16; % pixel with distance from center d > rPos are given a negative label
    opts.loss.rNeg = 0; % if rNeg != 0 pixels rPos < d < rNeg are given a neutral label
    opts.loss.labelWeight = 'balanced';
    
    opts.numPairs =  5.32e4; % Number of example pairs per epoch, if empty, then equal to number of videos.
    opts.randomSeed = 0;
    opts.shuffleDataset = false; % do not shuffle the data to get reproducible experiments
    opts.frameRange = 100; % range from the exemplar in which randomly pick the instance
    opts.gpus = 0;
    opts.prefetch = false; % Both get_batch and cnn_train_dag depend on prefetch.
    
    opts.train.numEpochs = 50;
    opts.train.learningRate = logspace(-5, -7, opts.train.numEpochs);
    opts.train.weightDecay = 5e-4;
    opts.train.batchSize = 8; % we empirically observed that small batches work better
    opts.train.profile = false;
    
    % Data augmentation settings
    opts.subMean = false;
    opts.colorRange = 255;
    opts.augment.translate = true;
    opts.augment.maxTranslate = 4;
    opts.augment.stretch = true;
    opts.augment.maxStretch = 0.05;
    opts.augment.color = true;
    opts.augment.grayscale = 0; % likelihood of using grayscale pair
 
end

