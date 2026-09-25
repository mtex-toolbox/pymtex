% the operations of bench/maps_ops.py in MATLAB
function bench_maps_ops(out, ops, K)

A = reshape(fread(fopen(fullfile(out,'map.bin')),'double'),3,[]).'; fclose('all');
cs = crystalSymmetry('m-3m','mineral','test');
ebsd = EBSD(vector3d(A(:,1),A(:,2),0), rotation.byAxisAngle(zvector,A(:,3)), ones(size(A,1),1), {cs}, struct());
xy = [ebsd.pos.x(:) ebsd.pos.y(:)];
grains = []; c = [];

% nested functions, since an anonymous one would keep the empty grains it was made with
names = {'calcgrains','perimeter','aspect','smooth_boundary','kam','grod','smooth_mean','smooth_spline'};
T = [names.', cellfun(@(n) @() operation(n), names, 'UniformOutput', false).', ...
  cellfun(@(n) @(res) rows(n, res), names, 'UniformOutput', false).'];
algos = {'grain segmentation','grain shape','boundary smoothing','intragranular misorientation','orientation denoising'};
crosscheckRun(out, T, ops, containers.Map(names, algos([1 2 2 3 4 4 5 5])), K, {'calcgrains'});

  function res = operation(name)
    switch name
      case 'calcgrains'
        [grains, ebsd.grainId] = calcGrains(ebsd, 'angle', 5*degree);
        c = [grains.centroid.x(:) grains.centroid.y(:)];
        res = grains;
      case 'perimeter', res = grains.perimeter;
      case 'aspect', res = grains.aspectRatio;
      case 'smooth_boundary', res = smoothBoundary(grains, 5);
      case 'kam', res = ebsd.KAM;
      case 'grod', res = angle(ebsd.calcGROD(grains));
      case 'smooth_mean', res = smooth(ebsd, meanFilter);
      case 'smooth_spline', res = smooth(ebsd, splineFilter);
    end
  end

  function R = rows(name, res)
    switch name
      case {'calcgrains','smooth_boundary'}
        if strcmp(name,'calcgrains'), v = res.area; else, v = res.perimeter; end
        R = [c v(:)];
      case {'perimeter','aspect'}, R = [c res(:)];
      case {'kam','grod'}, R = [xy res(:)];
      % a smoothed map comes back on its grid, in the grid's order
      otherwise, R = [res.pos.x(:) res.pos.y(:) res.rotations.a(:) res.rotations.b(:) res.rotations.c(:) res.rotations.d(:)];
    end
  end

end
