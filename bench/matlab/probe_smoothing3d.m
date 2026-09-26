function probe_smoothing3d()
% the boundary smoothing of mtex/maps3d/smoothing3d.py on the IN100 grains
e = EBSD3.load(fullfile(mtexDataPath,'EBSD','SmallIN100_MeshStats.dream3d'));
g = calcGrains(e,'angle',10*degree);
r = reduceBoundary(g); v = r.volume;
fprintf('reduce: %d faces, %d vertices, volume %.8f, 1:3 %s\n', size(r.F,1), length(r.boundary.V), sum(v), mat2str(v(1:3).',8));
fprintf('  F 1:2 %s\n', mat2str(r.F(1:2,:)));
s = smoothBoundary(r, taubinFilter(10)); v = s.volume;
fprintf('taubin: volume %.8f, 1:3 %s, area %.6f\n', sum(v), mat2str(v(1:3).',8), sum(s.boundary.area));
l = smoothBoundary(g, 3); v = l.volume;
fprintf('laplace3: volume %.8f, 1:3 %s, area %.6f\n', sum(v), mat2str(v(1:3).',8), sum(l.boundary.area));
q = reduceBoundary(g,'quadric'); v = q.volume;
fprintf('quadric: %d faces, volume %.8f, 1:3 %s\n', size(q.F,1), sum(v), mat2str(v(1:3).',8));
f = refineBoundary(g); fprintf('refine: %d faces, %d vertices\n', size(f.F,1), length(f.boundary.allV));
end
