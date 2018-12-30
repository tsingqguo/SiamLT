function [p] = initParams
    % These are the default hyper-params for SiamFC-3S
    % The ones for SiamFC (5 scales) are in params-5s.txt
    p.numScale = 3;
    p.scaleStep = 1.0375;
    p.scalePenalty = 0.9745;
    p.scaleLR = 0.59; % damping factor for scale update
    p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
    p.windowing = 'cosine'; % to penalize large displacements
    p.wInfluence = 0.176; % windowing influence (in convex sum)
    p.net = '2016-08-17.net.mat';

    %% execution, visualization, benchmark
    p.video = 'vot15_bag';
    p.visualization = false;
    p.gpus = 1;
    p.bbox_output = false;
    p.fout = -1;

    %% Params from the network architecture, have to be consistent with the training
    p.exemplarSize = 63;  % input z size
    p.instanceSize = 127;  % input x size (search region)
    p.scoreSize = 31;
    p.totalStride = 4;
    p.contextAmount = 0.5; % context amount for the exemplar
    p.subMean = false;

    %%feature setup
    % global feature parameter
    p.feat_global.normalize_power = 2;
    p.feat_global.normalize_size  = true;
    p.feat_global.normalize_dim   = true;
    p.feat_global.square_root_normalization = false;
    p.feat_global.cell_size = 4;
    p.feat_global.num_channel = 31;
    
    % selected features
    hog_params.cell_size = p.feat_global.cell_size;
    hog_params.num_channel = 31;
    
    cn_params.tablename = 'CNnorm';
    cn_params.useForGray = false;
    cn_params.cell_size = p.feat_global.cell_size;

    p.features = {
        struct('getFeature',@get_fhog, 'fparams',hog_params),...
%         struct('getFeature',@get_table_feature, 'fparams',cn_params),...    
    };
    
    %%
    p.framepaths = [];
    p.video = [];
    p.framepaths = [];
    p.init_rect = [];
    p.visualization = 0;
    p.gpus = 0;
    p.is_color_image =1;
    
    %% network setup
    p.id_feat_z = 'exemplar';
    p.id_feat_x = 'instance';
    p.id_score  = 'score';
end

