% -------------------------------------------------------------------------------------------------------------------------
function [newTargetPosition, bestScale] = tracker_eval(net,s_x, z_features, x_features, targetPosition, window, p)
%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------

    % forward pass, calculate response via correlation
    net.eval({p.id_feat_z, z_features,p.id_feat_x, x_features});
    scoreId = net.getVarIndex(p.id_score);
    responseMaps = reshape(net.vars(scoreId).value, [p.scoreSize p.scoreSize p.numScale]);
    responseMapsUP = single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, p.numScale));
    
    % Choose the scale whose response map has the highest peak
    if p.numScale>1
        currentScaleID = ceil(p.numScale/2);
        bestScale = currentScaleID;
        bestPeak = -Inf;
        for s=1:p.numScale
            if p.responseUp > 1
                % upsample to improve accuracy
                responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
            else
                responseMapsUP(:,:,s) = responseMaps(:,:,s);
            end
            thisResponse = responseMapsUP(:,:,s);
            % penalize change of scale
            if s~=currentScaleID, thisResponse = thisResponse * p.scalePenalty; end
            thisPeak = max(thisResponse(:));
            if thisPeak > bestPeak, bestPeak = thisPeak; bestScale = s; end
        end
        responseMap = responseMapsUP(:,:,bestScale);
    else
        responseMap = responseMapsUP;
        bestScale = 1;
    end
    
    % make the response map sum to 1
    responseMap = responseMap - min(responseMap(:));
    responseMap = responseMap / sum(responseMap(:));
    % apply windowing
    responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
    [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    p_corr = [r_max, c_max];
    % Convert to crop-relative coordinates to frame coordinates
    % displacement from the center in instance final representation ...
    disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
    % ... in instance input ...
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % ... in instance original crop (in frame coordinates)
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % position within frame in frame coordinates
    newTargetPosition = targetPosition + disp_instanceFrame;
end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end
