function probe_ebsd3()
% the volumes of mtex/maps3d/ebsd3.py and mtex/io/load3d.py: voxels, properties, one slice
e = EBSD3.load(fullfile(mtexDataPath,'EBSD','SmallIN100_MeshStats.dream3d'));
fprintf('dream3d %s size %s\n', class(e), mat2str(size(e)));
for ijk = [1 1 1; 50 30 70; 100 100 100; 13 87 42].'
  o = e(ijk(1),ijk(2),ijk(3));
  fprintf('  (%d,%d,%d) pos %.4f %.4f %.4f phase %d grainId %d q %.8f %.8f %.8f %.8f\n', ijk, o.pos.x, o.pos.y, o.pos.z, ...
    o.phaseId-1, o.grainId, o.rotations.a, o.rotations.b, o.rotations.c, o.rotations.d);
end
fprintf('  d1 %s d2 %s d3 %s\n', mat2str(e.d1.xyz,6), mat2str(e.d2.xyz,6), mat2str(e.d3.xyz,6));
s = e.slice(plane3d(vector3d(0,0,1), vector3d(0,0,-20)));
fprintf('  slice z=-20: %s, %d indexed, grainId sum %d\n', mat2str(size(s)), nnz(s.isIndexed), sum(s.grainId(~isnan(s.grainId))));
s = e.slice(plane3d(vector3d(1,1,1), vector3d(-25,22,-17)));
fprintf('  slice (1,1,1): %s, %d indexed\n', mat2str(size(s)), nnz(s.isIndexed));
x = EBSD3.load(fullfile(mtexDataPath,'EBSD3','SimulatedMultiPhase.h5'));
fprintf('xnovo %s size %s phases %d\n', class(x), mat2str(size(x)), numel(x.CSList));
for ijk = [1 1 1; 20 30 40; 50 50 50].'
  o = x(ijk(1),ijk(2),ijk(3));
  fprintf('  (%d,%d,%d) pos %.5f %.5f %.5f phase %d q %.8f %.8f %.8f %.8f\n', ijk, o.pos.x, o.pos.y, o.pos.z, ...
    o.phaseId-1, o.rotations.a, o.rotations.b, o.rotations.c, o.rotations.d);
end
for k = 2:numel(x.CSList), fprintf('  %s %s\n', x.CSList(k).mineral, x.CSList(k).pointGroup); end
end
