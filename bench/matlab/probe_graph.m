% the refined OR of the GrainGraphBasedReconstruction page, as a quaternion
warning('off','all'); rmpath(genpath('/home/hielscher/mtex/master')); warning('on','all');
mtexdata martensite silent
[grains,ebsd] = calcGrains(ebsd,'angle',3*degree,'minPixel',5,'alpha',6.1);
grains = smoothBoundary(grains,5);
job = parentGrainReconstructor(ebsd,grains);
job.p2c = orientation.KurdjumovSachs(job.csParent,job.csChild);
job.calcParent2Child;
r = job.p2c; fprintf('p2c %s\n', mat2str([r.a r.b r.c r.d],12));
