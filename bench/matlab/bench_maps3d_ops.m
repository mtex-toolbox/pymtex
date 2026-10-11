% the operations of bench/maps3d_ops.py in MATLAB
function bench_maps3d_ops(out, ops, K, in100, xnovo)

stride = struct('in100', 7, 'xnovo', 3);
ebsd = []; sim = [];
% a version without the 3D readers leaves the ops that use them out
if any(endsWith(ops, 'in100')), try, ebsd = EBSD3.load(in100); catch, end, end
if any(strcmp(ops, 'calcgrains3d_sim')), try, sim = EBSD3.load(xnovo); catch, end, end

% nested functions, since an anonymous one would keep the empty map it was made with
names = {'load_dream3d','load_xnovo','calcgrains3d_in100','calcgrains3d_sim','kam3d_in100'};
T = [names.', cellfun(@(n) @() operation(n), names, 'UniformOutput', false).', ...
  cellfun(@(n) @(res) rows(n, res), names, 'UniformOutput', false).'];
algos = {'3D EBSD import','3D grain segmentation','3D intragranular misorientation'};
% the KAM runs within the grains, so the segmentation runs as well when it is asked for
always = {};
if any(strcmp(ops, 'kam3d_in100')), always{end+1} = 'calcgrains3d_in100'; end
crosscheckRun(out, T, ops, containers.Map(names, algos([1 1 2 2 3])), K, always);

  function res = operation(name)
    switch name
      case 'load_dream3d', res = EBSD3.load(in100);
      case 'load_xnovo', res = EBSD3.load(xnovo);
      case 'calcgrains3d_in100', [res, ebsd] = calcGrains(ebsd, 'angle', 5*degree);
      case 'calcgrains3d_sim', res = calcGrains(sim, 'angle', 5*degree);
      case 'kam3d_in100', res = ebsd.KAM;
    end
  end

  function R = rows(name, res)
    switch name
      case 'load_dream3d', R = voxelRows(res, stride.in100);
      case 'load_xnovo', R = voxelRows(res, stride.xnovo);
      case {'calcgrains3d_in100','calcgrains3d_sim'}
        c = res.centroid; v = res.volume;
        R = [c.x(:) c.y(:) c.z(:) v(:) res.numPixel(:)];
      case 'kam3d_in100'
        [keep, xyz] = sampled(ebsd, stride.in100);
        R = [xyz(keep,:) res(keep)];
    end
  end

end


% the voxels whose lattice indices along every axis are multiples of the stride
function [keep, xyz] = sampled(ebsd, stride)
xyz = [ebsd.pos.x(:) ebsd.pos.y(:) ebsd.pos.z(:)];
keep = true(size(xyz,1),1);
for d = 1:3
  [~,~,i] = unique(round(xyz(:,d), 9));
  keep = keep & mod(i-1, stride) == 0;
end
end


% position, phase counted from zero with 0 the notIndexed as in Python, and the quaternion
function R = voxelRows(ebsd, stride)
[keep, xyz] = sampled(ebsd, stride);
q = ebsd.rotations;
R = [xyz ebsd.phaseId(:)-1 q.a(:) q.b(:) q.c(:) q.d(:)];
R = R(keep,:);
end
