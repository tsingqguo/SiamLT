% -------------------------------------------------------------------------------------------------------------------------
classdef XCorrf < dagnn.Layer
%XCORR
%   Crosscorrelates two activations of different size exploiting  the API of vl_nnconv
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016, modified by
%   Qing Guo
% -------------------------------------------------------------------------------------------------------------------------
    properties
        opts = {'cuDNN'}
    end

    methods
        function outputs = forward(obj, inputs, params)
            assert(numel(inputs) == 2, 'two inputs are needed');

            z = inputs{1}; % exemplar
            x = inputs{2}; % instance (search region)

            assert(ndims(z) <= ndims(x), 'z and x have different number of dimensions');
            assert(size(z,1) <= size(x,1), 'exemplar z has to be smaller than instance x');
            
            % padding z to 
            [wx,hx,cx,bx] = size(x);
            [wz,hz,cz,bz] = size(z);
            
            p_z = zeros(hx,wx,cz,bz,'single');
            h_pzidxL = floor((hx-hz)./2+1);
            h_pzidxH = h_pzidxL+hz+1;
            w_pzidxL = floor((wx-wz)./2+1);
            w_pzidxH = w_pzidxL+wz+1;
            p_z(h_pzidxL+1:h_pzidxH-1,w_pzidxL+1:w_pzidxH-1,1:cz,1:bz) = z;

            p_zf = fft2(p_z);
            xf = fft2(x);
            of= sum(p_zf.*xf,3);
            
            o = real(fftshift(ifft2(of)));
            outputs{1} = o;%o(h_pzidxL+1:h_pzidxH-1,w_pzidxL+1:w_pzidxH-1,1:cz,1:bz);
            
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            assert(numel(inputs) == 2, 'two inputs are needed');
            assert(numel(derOutputs) == 1, 'only one gradient should be flowing in this layer (dldy)');
            z = inputs{1}; % exemplar
            x = inputs{2}; % instance
            assert((size(z,1) < size(x,1)) ||(size(z,2) < size(x,2)), 'exemplar z has to be smaller than instance x');
            [wx,hx,cx,bx] = size(x);
            [wz,hz,cz,bz] = size(z);
            p_z = zeros(hx,wx,cz,bz,'single');
            h_pzidxL = floor((hx-hz)./2+1);
            h_pzidxH = h_pzidxL+hz+1;
            w_pzidxL = floor((wx-wz)./2+1);
            w_pzidxH = w_pzidxL+wz+1;            
            p_z(h_pzidxL+1:h_pzidxH-1,w_pzidxL+1:w_pzidxH-1,1:cz,1:bz) = z;
            
            p_zf = fft2(p_z);
            xf = fft2(x);
            
            dldy = derOutputs{1};
            [wdl,hdl,cdl,bdl] = size(dldy);

            dldyf = fft2(dldy);
            dldzf = xf.*repmat(dldyf,1,1,size(xf,3),1);
            dldxf = p_zf.*repmat(dldyf,1,1,size(xf,3),1);
            
            dldz = real(ifft2(dldzf));
            dldx = real(ifft2(dldxf));
            
            [mx,nx,cb,one] = size(dldx);
            assert(mx == size(x, 1));
            assert(nx == size(x, 2));
            assert(cb == cx);
            assert(one == bx);
            derInputs{1} = dldz(h_pzidxL+1:h_pzidxH-1,w_pzidxL+1:w_pzidxH-1,1:cz,1:bz);
            derInputs{2} = dldx;
            derParams = {};
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            z_sz = inputSizes{1};
            x_sz = inputSizes{2};
            y_sz = [x_sz(1:2),1, z_sz(4)];
            outputSizes = {y_sz};
        end

        function rfs = getReceptiveFields(obj)
            rfs(1,1).size = [inf inf]; % could be anything
            rfs(1,1).stride = [1 1];
            rfs(1,1).offset = 1;
            rfs(2,1).size = [inf inf];
            rfs(2,1).stride = [1 1];
            rfs(2,1).offset = 1;
        end

        function obj = XCorr(varargin)
            obj.load(varargin);
        end

    end

end
