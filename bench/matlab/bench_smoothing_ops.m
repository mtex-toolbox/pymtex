% the operations of bench/smoothing_ops.py in MATLAB
function bench_smoothing_ops(out, ops, K)

A = reshape(fread(fopen(fullfile(out,'map.bin')),'double'),4,[]).'; fclose('all');
cs = crystalSymmetry('m-3m','mineral','test');
ebsd = EBSD(vector3d(A(:,1),A(:,2),0), rotation.byAxisAngle(zvector,A(:,3)), A(:,4)+1, ...
  {'notIndexed', cs}, struct(), 'phaseMap', [0 1]);
ebsd = ebsd.gridify;
grains = [];

names = {'staircase','smooth','smooth_noSimplify','simplify_staircase'};
T = [names.', cellfun(@(n) @() operation(n), names, 'UniformOutput', false).', ...
  repmat({@inner}, numel(names), 1)];
crosscheckRun(out, T, ops, containers.Map(names, {'grain segmentation','boundary smoothing', ...
  'boundary smoothing','boundary smoothing'}), K, {'staircase'});

  % nested, since an anonymous function would keep the empty grains it was made with
  function res = operation(name)
    switch name
      case 'staircase'
        grains = calcGrains(ebsd, 'angle', 5*degree, 'straighten', false);
        res = grains;
      case 'smooth', res = smoothBoundary(grains);
      case 'smooth_noSimplify', res = smoothBoundary(grains, 'noSimplify');
      case 'simplify_staircase', res = simplifyBoundary(grains);
    end
  end

end


% the segments of both boundaries between two grains
function R = inner(g)
R = [rows(g.boundary); rows(g.innerBoundary)];
end

function R = rows(gB)
V = [gB.allV.x(:) gB.allV.y(:)];
ok = all(gB.grainId > 0, 2);
R = [V(gB.F(ok,1),:) V(gB.F(ok,2),:)];
end
