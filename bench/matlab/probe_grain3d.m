function probe_grain3d()
% the measures and sections of mtex/maps3d/grain3d.py and section3d.py on the IN100 grains
e = EBSD3.load(fullfile(mtexDataPath,'EBSD','SmallIN100_MeshStats.dream3d'));
g = calcGrains(e,'angle',10*degree);
d = g.diameter; fprintf('diameter 1:5 %s\n', mat2str(d(1:5).',8));
% master fails on triangles (c = c.xyz on a double), so the triangles go in as closed polygons
gc = g; gc.boundary.F = num2cell(double([g.F g.F(:,1)]),2);
[a,b,c] = gc.principalComponents;
fprintf('pc |a| 1:5 %s\n|b| %s\n|c| %s\n', mat2str(norm(a(1:5)).',8), mat2str(norm(b(1:5)).',8), mat2str(norm(c(1:5)).',8));
fprintf('pc |a| 700:705 %s\n', mat2str(norm(a(700:705)).',8));
p = g.neighbors; fprintf('neighbors %d, numNeighbors 1:10 %s\n', size(p,1), mat2str(subsref(numNeighbors(g),substruct('()',{1:10})).'));
pe = g.boundary.perimeter; fprintf('perimeter 1:3 %s sum %.6f\n', mat2str(pe(1:3).',8), sum(pe));
t = g.boundary.nodeType; fprintf('nodeType hist %s\n', mat2str(accumarray(t+1,1).'));
E = g.boundary.edges; fprintf('edges %d\n', size(E,1));
s = g(g.numPixel>1000); fprintf('subset %d grains, %d faces\n', length(s), size(s.F,1));
pl = plane3d(vector3d(0,0,1), vector3d(0,0,-20));
fprintf('intersected %d\n', nnz(g.intersected(pl)));
s2 = g.slice(pl); fprintf('slice z: %d grains, area %.6f, Id3d %s\n', length(s2), sum(s2.area), mat2str(sort(s2.prop.Id3d(:)).'));
s3 = g.slice(plane3d(vector3d(1,1,1), vector3d(-25,22,-17))); fprintf('slice 111: %d grains, area %.6f\n', length(s3), sum(s3.area));
fprintf('perimeter of slice z %.6f\n', sum(s2.perimeter));
end
