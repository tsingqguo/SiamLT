% ----------------------------------------------------------------------------------------------------------------
function net = vid_create_net_HOGConv(varargin)
% ----------------------------------------------------------------------------------------------------------------
    opts.exemplarSize = [127 127];
    opts.instanceSize = [255 255];
    opts.num_Channel = 31;
    opts.scale = 1 ;
    opts.initBias = 0.1 ;
    opts.weightDecay = 1 ;
    %opts.weightInitMethod = 'xavierimproved' ;
    opts.weightInitMethod = 'gaussian';
    opts.batchNormalization = false ;
    opts.networkType = 'simplenn' ;
    opts.strides = [1,1,1] ;
    opts.cudnnWorkspaceLimit = 1024*1024*1024 ; % 1GB
    opts = vl_argparse(opts, varargin) ;

    if numel(opts.exemplarSize) == 1
        opts.exemplarSize = [opts.exemplarSize, opts.exemplarSize];
    end
    
    if numel(opts.instanceSize) == 1
        opts.instanceSize = [opts.instanceSize, opts.instanceSize];
    end

    net = create_hogconv_net(struct(), opts) ;

    % Meta parameters
    net.meta.normalization.interpolation = 'bicubic' ;
    net.meta.normalization.averageImage = [] ;
    net.meta.normalization.keepAspect = true ;
    net.meta.augmentation.rgbVariance = zeros(0,3) ;
    net.meta.augmentation.transformation = 'stretch' ;

    % Fill in default values
    net = vl_simplenn_tidy(net) ;
end

% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;

if fc
  name = 'fc' ;
else
  name = 'conv' ;
end

convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate',[1 1],...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
                       
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), zeros(out, 2, 'single')}}, ...
                             'learningRate', [2 1 0.05], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

end

% --------------------------------------------------------------------
function net = add_block_conv_only(net, opts, id, h, w, in, out, stride, pad, init_bias)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate',[1 1],...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
end

% --------------------------------------------------------------------
function net = add_norm(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'normalize', ...
                             'name', sprintf('norm%s', id), ...
                             'param', [5 1 0.0001/5 0.75]) ;
end
end

% --------------------------------------------------------------------
function net = add_dropout(net, opts, id)
% --------------------------------------------------------------------
if ~opts.batchNormalization
  net.layers{end+1} = struct('type', 'dropout', ...
                             'name', sprintf('dropout%s', id), ...
                             'rate', 0.5) ;
end
end

% --------------------------------------------------------------------
function net = create_hogconv_net(net, opts)
% --------------------------------------------------------------------
    
    strides = ones(1, 3);
    strides(1:numel(opts.strides)) = opts.strides(:);

    net.layers = {} ;

    net = add_block(net, opts, '1', 3, 3, opts.num_Channel, 96, strides(1), 0) ;
    net = add_block(net, opts, '2', 3, 3, 96, 256, strides(2), 0) ;
    net = add_block_conv_only(net, opts, '3', 3, 3, 256, 256, strides(3), 0) ;

    % Check if the receptive field covers full image
    [ideal_exemplar, ~] = ideal_size(net, opts.exemplarSize);
    [ideal_instance, ~] = ideal_size(net, opts.instanceSize);
    assert(sum(opts.exemplarSize==ideal_exemplar)==2, 'exemplarSize is not ideal.');
    assert(sum(opts.instanceSize==ideal_instance)==2, 'instanceSize is not ideal.');
end
