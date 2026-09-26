function probe_mcl()
% the grain counts of doc/Grains/GrainReconstructionMCL.m, for the MCL segmentation of mtex/maps/segment.py
ebsd = mtexdata('EMSphinx','silent');
ebsd = ebsd('Iron fcc');
ebsd = ebsd(inpolygon(ebsd,[40 30 80 60]));
fprintf('pixels %d\n', length(ebsd));
g = calcGrains(ebsd,'angle',10*degree,'minPixel',10); fprintf('hard: %d inner %d\n', length(g), length(g.innerBoundary));
g = calcGrains(ebsd,'mcl',[1.2 4],'angle',10*degree,'minPixel',10); fprintf('hard + MCL: %d inner %d\n', length(g), length(g.innerBoundary));
g = calcGrains(ebsd,'soft',[1 0.5]*degree,'minPixel',10); fprintf('soft: %d inner %d\n', length(g), length(g.innerBoundary));
g = calcGrains(ebsd,'mcl',[1.2 4],'soft',[1 0.5]*degree,'minPixel',10); fprintf('soft + MCL: %d inner %d\n', length(g), length(g.innerBoundary));
for p = [1.1 1.4 1.8]
  g = calcGrains(ebsd,'mcl',[p 4],'soft',[1 0.5]*degree,'minPixel',10); fprintf('inflation %.1f: %d\n', p, length(g));
end
end
