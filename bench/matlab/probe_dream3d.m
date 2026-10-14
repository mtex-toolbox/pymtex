function probe_dream3d(fname)
% the DREAM.3D voxel import of mtex/io/load3d.py on a file of any layout: voxels, phases, properties
tic; e = EBSD3.load(fname); t = toc;
fprintf('%s size %s in %.1f s\n', class(e), mat2str(size(e)), t);
fprintf('  dxyz %s phases %s\n', mat2str([e.dx e.dy e.dz],6), mat2str(accumarray(e.phaseId(:),1).'));
for k = 1:numel(e.CSList)
  cs = e.CSList(k);
  if isa(cs,'crystalSymmetry'), fprintf('  %d %s\n', k-1, cs.pointGroup); else, fprintf('  %d notIndexed\n', k-1); end
end
sz = size(e);
for ijk = [1 1 1; sz; ceil(sz/2); ceil(sz/3) + [3 -2 1]].'
  o = e(ijk(1),ijk(2),ijk(3));
  fprintf('  (%d,%d,%d) pos %.5f %.5f %.5f phase %d q %.8f %.8f %.8f %.8f\n', ijk, o.pos.x, o.pos.y, o.pos.z, ...
    o.phaseId-1, o.rotations.a, o.rotations.b, o.rotations.c, o.rotations.d);
end
names = sort(fieldnames(e.prop));
for k = 1:numel(names)
  v = e.prop.(names{k});
  fprintf('  %s %.10g\n', names{k}, sum(double(v(isfinite(v)))));
end
end
