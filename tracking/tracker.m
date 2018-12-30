% -------------------------------------------------------------------------------------------------
function [bboxes,fps] = tracker(varargin)
%TRACKER
%   is the main function that performs the tracking loop
%   Default parameters are overwritten by VARARGIN
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------
    % initialize params
    p = initParams();
    % Overwrite default parameters with varargin
    p = vl_argparse(p, varargin);
    
% -------------------------------------------------------------------------------------------------
    % Get environment-specific default paths.
    p = env_paths_tracking(p);

    % initialize a network
    net = init_net();
    % Load sequences information
    [imgFiles, targetPosition, targetSize] = load_video_info(p.framepaths, p.init_rect);
    nImgs = numel(imgFiles);
    startFrame = 1;
    % get the first frame of the video
    im = uint8(imgFiles{startFrame});
    % if grayscale repeat one channel to match filters size
	if(size(im, 3)==1)
        im = repmat(im, [1 1 3]);
    end
    avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);
    
    % Init visualization
    videoPlayer = [];
%     if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
%         videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
%     end
    
    %
    wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
    hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
    s_z = sqrt(wc_z*hc_z);
    scale_z = p.exemplarSize / s_z;
    
    % initialize the exemplar
    [z_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans);  
    d_search = (p.instanceSize - p.exemplarSize)/2;
    pad = d_search/scale_z;
    s_x = s_z + 2*pad;
    
    % arbitrary scale saturation
    min_s_x = 0.2*s_x;
    max_s_x = 5*s_x;

    switch p.windowing
        case 'cosine'
            window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
        case 'uniform'
            window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
    end
    
    % make the window sum 1
    window = window / sum(window(:));
    scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));
    
    % extract features of exemplar z
    [p.features] = ...
        init_features(p.features,p.is_color_image,[round(s_z) round(s_z)],[round(s_x) round(s_x)], 'odd_cells');
    z_features = extract_features(z_crop,p.features, p.feat_global,'z',p.numScale);

    bboxes = zeros(nImgs, 4);
    % start tracking
    time = 0;
    
    for i = startFrame:nImgs
        tic;
        if i>startFrame
            % load a new frame
            im = uint8(imgFiles{i});
   			% if grayscale repeat one channel to match filters size
    		if(size(im, 3)==1)
        		im = repmat(im, [1 1 3]);
    		end
            scaledInstance = s_x .* scales;
            scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];          
            % extract scaled crops for search region x at previous target position
            x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans,p);
            % extract feature of x_crops
            x_features = extract_features(x_crops,p.features, p.feat_global,'x',1);
            % evaluate the offline-trained network for exemplar x features
            [newTargetPosition, newScale] = tracker_eval(net, round(s_x), z_features, x_features, targetPosition, window, p);
            
            targetPosition = gather(newTargetPosition);
            % scale damping and saturation
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
        else
            % at the first frame output position and size passed as input (ground truth)
        end

        rectPosition = [targetPosition([2,1]) - targetSize([2,1])/2, targetSize([2,1])];
        % output bbox in the original frame coordinates
        oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
        oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
        bboxes(i, :) = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];
        time = time+toc; 
        
        if p.visualization
            if isempty(videoPlayer)
                figure(1), imshow(double(im)/255);
                figure(1), rectangle('Position', rectPosition, 'LineWidth', 4, 'EdgeColor', 'y');
                drawnow
                fprintf('Frame %d\n', startFrame+i);
            else
                im = gather(im)/255;
                im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 4, 'Color', 'yellow');
                % Display the annotated video frame using the video player object.
                step(videoPlayer, im);
            end
        end
        if p.bbox_output
            fprintf(p.fout,'%.2f,%.2f,%.2f,%.2f\n', bboxes(i, :));
        end
    end
    
    bboxes = bboxes(startFrame : i, :);
    fps = nImgs./time;
end
