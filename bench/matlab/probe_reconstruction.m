% the MaParentGrainReconstruction steps with the numbers bench/reconstruction.py prints
warning('off','all'); rmpath(genpath('/home/hielscher/mtex/master')); warning('on','all');
mtexdata martensite silent
tic; [grains,ebsd] = calcGrains(ebsd,'angle',3*degree,'minPixel',2,'alpha',12); fprintf('calcGrains %.2f s %d\n', toc, length(grains));
job = parentGrainReconstructor(ebsd,grains);
job.p2c = orientation.KurdjumovSachs(job.csParent,job.csChild);
tic; f = job.calcGBFit./degree; fprintf('calcGBFit %.2f s %s\n', toc, mat2str(quantile(f,[.2 .4 .6 .8]),4));
tic; job.calcParent2Child; fprintf('calcParent2Child %.2f s %.4f\n', toc, angle(job.p2c, orientation.KurdjumovSachs(job.csParent,job.csChild))./degree);
f = job.calcGBFit./degree; fprintf('fit %s\n', mat2str(quantile(f,[.2 .4 .6 .8]),4));
disp(char(job.p2c)); r = job.p2c; fprintf('p2c %s\n', mat2str([r.a r.b r.c r.d],10));
tic; job.calcVariantGraph('threshold',3.5*degree,'tolerance',1.5*degree); fprintf('calcVariantGraph %.2f s %d\n', toc, nnz(job.graph));
tic; job.clusterVariantGraph('includeSimilar'); fprintf('clusterVariantGraph %.2f s\n', toc);
fprintf('votes %s\n', mat2str(quantile(job.votes.prob(:,1),[.2 .4 .6 .8]),4));
tic; job.calcParentFromVote('minProb',0.5); fprintf('calcParentFromVote %.2f s %d\n', toc, nnz(job.isParent));
tic; job.calcGBVotes('p2c','reconsiderAll','threshold',4*degree,'tolerance',1.5*degree); job.calcParentFromVote; fprintf('growth %.2f s %d\n', toc, nnz(job.isParent));
tic; job.mergeSimilar('threshold',7.5*degree); fprintf('mergeSimilar %.2f s %d\n', toc, length(job.grains));
tic; job.mergeInclusions('maxSize',50); fprintf('mergeInclusions %.2f s %d %d\n', toc, length(job.grains), nnz(job.isParent));
tic; job.calcVariants; fprintf('calcVariants %.2f s %s\n', toc, mat2str(histcounts(job.packetId,0.5:1:4.5)));
tic; pe = job.calcParentEBSD; fprintf('calcParentEBSD %.2f s %s\n', toc, mat2str(quantile(pe.fit./degree,[.2 .4 .6 .8]),4));
job
