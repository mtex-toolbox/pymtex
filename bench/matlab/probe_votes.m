% the vote and grain graph routes on the martensite map with MATLAB's refined OR, the
% numbers the second half of bench/reconstruction.py --routes prints
warning('off','all'); rmpath(genpath('/home/hielscher/mtex/master')); warning('on','all');
mtexdata martensite silent
[grains,ebsd] = calcGrains(ebsd,'angle',3*degree,'minPixel',2,'alpha',12);
job = parentGrainReconstructor(ebsd,grains);
job.p2c = orientation(rotation(quaternion(0.9183690176,-0.01211658611,0.3880635769,0.07653885347)),job.csParent,job.csChild);
tic; job.calcGBVotes; fprintf('calcGBVotes %.2f s %s\n', toc, mat2str(quantile(job.votes.prob(:,1),[.2 .4 .6 .8]),4));
tic; job.calcTPVotes; fprintf('calcTPVotes %.2f s %s\n', toc, mat2str(quantile(job.votes.prob(:,1),[.2 .4 .6 .8]),4));
tic; job.calcGraph('threshold',2.5*degree,'tolerance',2.5*degree); fprintf('calcGraph %.2f s %d\n', toc, nnz(job.graph));
tic; job.clusterGraph; fprintf('clusterGraph %.2f s\n', toc);
if exist('full','var'), tic; job.calcParentFromGraph; fprintf('calcParentFromGraph %.2f s %d\n', toc, nnz(job.isParent)); end
