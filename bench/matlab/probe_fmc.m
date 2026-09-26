function probe_fmc(cmaha)
% the FMC hierarchy and grain count on the zoom of doc/Grains/GrainReconstruction.m, for mtex/maps/fmc.py
ebsd = mtexdata('EMSphinx','silent');
ebsd = ebsd('Iron fcc');
ebsd = ebsd(inpolygon(ebsd,[40 30 80 60]));
fprintf('pixels %d\n', length(ebsd));
tic; g = calcGrains(ebsd,'fmc',cmaha,'minPixel',10,'verbose'); t = toc;
fprintf('cmaha %g grains %d inner %d  %.2fs\n', cmaha, length(g), length(g.innerBoundary), t);
end
