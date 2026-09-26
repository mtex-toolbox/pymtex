function probe_resample()
% gridify(ebsd,'unitCell',uC) on the maps of doc/EBSDAnalysis/EBSDInter.m and EBSDGrid.m, for
% the resampling of mtex/maps/gridify.py: shape, indexed count, the first sites' positions and
% phases, and the mean misorientation to the Python orientations read back from a file
ebsd = mtexdata('twins','silent');
uC = rotate(2*ebsd.unitCell,45*degree);
report('twins', gridify(ebsd,'unitCell',uC));
ebsd = mtexdata('ferrite','silent');
sq = ebsd.dPos / 4 * vector3d([-1 -1 1 1],[-1 1 1 -1],0).';
report('ferrite', gridify(ebsd,'unitCell',sq));
ebsd = mtexdata('copper','silent');
report('copper25', gridify(ebsd,'unitCell',2.5 * vector3d([-1 -1 1 1].',[-1 1 1 -1].',0)));
report('copper05', gridify(ebsd,'unitCell',0.5 * vector3d([-1 -1 1 1].',[-1 1 1 -1].',0)));
end

function report(name, g)
x = g.pos.x(:); y = g.pos.y(:);
fprintf('%s: size %d x %d, indexed %d, x %.4f..%.4f, y %.4f..%.4f\n', name, size(g,1), size(g,2), nnz(g.isIndexed), min(x), max(x), min(y), max(y));
q = g.rotations(:);
fid = fopen(fullfile('/home/hielscher/mtex/pymtex/bench/out', ['resample_' name '.csv']), 'w');
fprintf(fid, '%.9g,%.9g,%d,%.9g,%.9g,%.9g,%.9g\n', [x, y, double(g.phaseId(:)), q.a(:), q.b(:), q.c(:), q.d(:)].');
fclose(fid);
end
