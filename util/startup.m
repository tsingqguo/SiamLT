addpath(genpath('/home/tsingqguo/Desktop/Object_tracking/tracking_benchmark_v1.0/tool/matconvnet/matlab'));
vl_setupnn;

[pathstr, ~, ~] = fileparts(mfilename('fullpath'));
% PDollar toolbox
addpath(genpath([pathstr '/external_libs/pdollar_toolbox/channels']));