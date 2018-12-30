function run_experiment(imdb_video)

	% Parameters that should have no effect on the result.
	opts.prefetch = false;
	opts.gpus = [];

	if nargin < 1
	    imdb_video = [];
	end
	experiment(imdb_video, opts);

end

