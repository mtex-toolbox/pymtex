function probe_calcgrains3d()
% the grains of mtex/maps3d/calcgrains3d.py on the IN100 volume
e = EBSD3.load(fullfile(mtexDataPath,'EBSD','SmallIN100_MeshStats.dream3d'));
for c = {{'angle',10*degree}, {'angle',5*degree,'minPixel',10}}
  tic; [g, e2] = calcGrains(e, c{1}{:}); t = toc;
  fprintf('%s: %d grains in %.2f s, %d faces, %d vertices\n', strjoin(cellfun(@num2str,c{1},'UniformOutput',false),' '), length(g), t, size(g.F,1), length(g.boundary.V));
  fprintf('  numPixel 1:10 %s\n', mat2str(g.numPixel(1:10).'));
  fprintf('  grainId at (1,1,1) (50,30,70) %d %d, max %d\n', e2.grainId(1,1,1), e2.grainId(50,30,70), max(e2.grainId(:)));
  v = g.volume; fprintf('  volume sum %.6f min %.6f\n', sum(v), min(v));
  fprintf('  inner faces %d\n', nnz(all(g.boundary.grainId>0,2)));
  q = g.prop.meanRotation(1:3);
  fprintf('  mean 1:3 %s\n', mat2str([q.a q.b q.c q.d],8));
  fprintf('  GOS 1:5 %s\n', mat2str(g.prop.GOS(1:5).'/degree,6));
  gB = g.boundary(all(g.boundary.grainId>0,2));
  a = angle(gB.misorientation); fprintf('  misorientation mean %.6f max %.6f (deg)\n', mean(a)/degree, max(a)/degree);
  c0 = g.centroid; fprintf('  centroid 1 %.6f %.6f %.6f\n', c0.x(1), c0.y(1), c0.z(1));
end
end
