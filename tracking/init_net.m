function [net] = init_net()

net = dagnn.DagNN();
net.addLayer('xcorr',XCorrf(),{'exemplar','instance'},{'xcorr_out'},{});
add_adjust_layer(net, 'adjust', 'xcorr_out', 'score', ...
                 {'adjust_f', 'adjust_b'}, 1e-3, 0, 0, 1);
net.mode = 'test';

end

