function results = run_SiamLT(seq, res_path, bSaveImage,varargin)
    
    %% setup pathes for 
    addpath(genpath('./tracking'));
    addpath(genpath('./utils'));
    addpath(genpath('./feature_extraction'));
    startup;
    
    %% Parameters that should have no effect on the result.
    params.video = seq.name(1:end-2);
    params.framepaths = seq.s_frames;
    params.init_rect = seq.init_rect;
    params.visualization = bSaveImage;
    params.gpus = 0;
    
    %% Call the main tracking function
    [rects,fps]=tracker(params); 
    results.type   = 'rect';
    results.res    = rects;
    results.fps    = fps;    
end