function probe_loadgrains3d()
% the grain meshes of mtex/io/loadgrains3d.py and mtex/maps3d/orientfaces.py
g = grain3d.load(fullfile(mtexDataPath,'EBSD','SmallIN100_MeshStats.dream3d'));
v = g.volume;
fprintf('dream3d: %d grains, %d faces, volume sum %.6f, negative %d, 1:5 %s\n', length(g), size(g.F,1), sum(v), nnz(v<0), mat2str(v(1:5).',8));
fprintf('  I_GF + per face %s\n', mat2str(accumarray(full(sum(g.I_GF,1)).'+2,1).'));
fprintf('  grainId 1:3 %s\n', mat2str(g.boundary.grainId(1:3,:)));
q = g.prop.meanRotation(1:2); fprintf('  mean as read 1:2 %s\n', mat2str([q.a q.b q.c q.d],6));
n = grain3d.load(fullfile(mtexDataPath,'Neper','my100grains.tess'));
v = n.volume;
fprintf('tess: %d grains, %d faces, volume sum %.6f, 1:5 %s\n', length(n), length(n.F), sum(v), mat2str(v(1:5).',8));
q = n.prop.meanRotation(1:2); fprintf('  mean 1:2 %s\n', mat2str([q.a q.b q.c q.d],6));
c = n.centroid; fprintf('  centroid 1 %.8f %.8f %.8f\n', c.x(1), c.y(1), c.z(1));
fprintf('  numNeighbors 1:10 %s\n', mat2str(subsref(numNeighbors(n),substruct('()',{1:10})).'));
d = n.diameter; fprintf('  diameter 1:5 %s\n', mat2str(d(1:5).',8));
s = n.slice(plane3d(vector3d(0,0,1),vector3d(0,0,1))); fprintf('  slice z=1: %d grains, area %.6f\n', length(s), sum(s.area));
end
