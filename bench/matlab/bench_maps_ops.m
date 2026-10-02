% the operations of bench/maps_ops.py in MATLAB
function bench_maps_ops(out, ops, K, large)

A = reshape(fread(fopen(fullfile(out,'map.bin')),'double'),3,[]).'; fclose('all');
cs = crystalSymmetry('m-3m','mineral','test');
ebsd = EBSD(vector3d(A(:,1),A(:,2),0), rotation.byAxisAngle(zvector,A(:,3)), ones(size(A,1),1), {cs}, struct());
xy = [ebsd.pos.x(:) ebsd.pos.y(:)];
grains = []; c = []; big = []; job = []; page = [];
if any(strcmp(ops,'calcgrains_1C_1')), big = loadNeutral(large); end
if any(startsWith(ops,'trueebsd')), page = trueEbsdPage; end
% the measured maps as the EBSDDenoising page prepares them
measured = struct;
for m = {'twins','ferrite'}
  if any(endsWith(ops, m{1})), measured.(m{1}) = prepare(m{1}); end
end

% nested functions, since an anonymous one would keep the empty grains it was made with
names = {'calcgrains_1C_1','calcgrains','perimeter','aspect','smooth_boundary','kam','grod', ...
  'smooth_mean_twins','smooth_spline_twins','smooth_l1tv_twins','smooth_hq_twins', ...
  'smooth_mean_ferrite','smooth_spline_ferrite','smooth_l1tv_ferrite','smooth_hq_ferrite', ...
  'smooth_fill_twins','smooth_fill_ferrite','trueebsd_distortion','trueebsd_undistort'};
T = [names.', cellfun(@(n) @() operation(n), names, 'UniformOutput', false).', ...
  cellfun(@(n) @(res) rows(n, res), names, 'UniformOutput', false).'];
algos = {'grain segmentation','grain shape','boundary smoothing','intragranular misorientation','orientation denoising','map registration'};
% the grains and the distortion job run as well only when an operation asked for needs them
always = {};
if any(ismember(ops, {'perimeter','aspect','smooth_boundary','kam','grod'})), always{end+1} = 'calcgrains'; end
if any(strcmp(ops, 'trueebsd_undistort')), always{end+1} = 'trueebsd_distortion'; end
crosscheckRun(out, T, ops, containers.Map(names, algos([1 1 2 2 3 4 4 5 5 5 5 5 5 5 5 5 5 6 6])), K, always, ...
  containers.Map({'calcgrains_1C_1','trueebsd_distortion','smooth_l1tv_twins','smooth_l1tv_ferrite','smooth_hq_twins','smooth_hq_ferrite'}, {1, 1, 1, 1, 1, 1}));

  function res = operation(name)
    switch name
      case 'calcgrains_1C_1'
        g = calcGrains(big, 'angle', 10*degree);
        a = g.area;
        res = [g.centroid.x(:) g.centroid.y(:) a(:)];
      case 'calcgrains'
        [grains, ebsd] = calcGrains(ebsd, 'angle', 5*degree);
        c = [grains.centroid.x(:) grains.centroid.y(:)];
        res = grains;
      case 'perimeter', res = grains.perimeter;
      case 'aspect', res = grains.aspectRatio;
      case 'smooth_boundary', res = smoothBoundary(grains, 5);
      case 'kam', res = ebsd.KAM;
      case 'grod', res = angle(ebsd.calcGROD(grains));
      % a fresh job on the matched grid, since calcDistortion alters the one it runs on
      case 'trueebsd_distortion'
        job = trueEbsd(page.imgList, page.hops());
        job.resizedList = page.resized;
        job.setOptions('numROI',16);
        job.calcDistortion('fitErr','silent');
        res = job;
      case 'trueebsd_undistort'
        job.undistort;
        res = job.undistortedList(1).ebsd;
      otherwise
        parts = split(name, '_');
        filters = struct('mean', @meanFilter, 'spline', @splineFilter, 'l1tv', @l1TVFilter, 'hq', @l1TVFilter, 'fill', @splineFilter);
        if strcmp(parts{2}, 'fill'), flag = {'fill'}; else, flag = {}; end
        res = smooth(measured.(parts{3}), filters.(parts{2})(), flag{:});
        res = res('indexed');
    end
  end

  function R = rows(name, res)
    switch name
      case 'calcgrains_1C_1', R = res;
      case {'calcgrains','smooth_boundary'}
        if strcmp(name,'calcgrains'), v = res.area; else, v = res.perimeter; end
        R = [c v(:)];
      case {'perimeter','aspect'}, R = [c res(:)];
      case {'kam','grod'}, R = [xy res(:)];
      % the tiles as (hop, stage, centre, shift, peak), counted from zero as in Python, stage -1 the fitError pass
      case 'trueebsd_distortion'
        R = zeros(0,7);
        for n = 1:numel(res.shifts)
          for m = 1:numel(res.shifts{n})
            ps = res.shifts{n}(m);
            if ~isempty(ps.u), R = [R; tileRows(n-1, m-1, ps)]; end %#ok<AGROW>
          end
        end
        for n = 1:numel(res.fitError)
          ps = res.fitError(n);
          if ~isempty(ps.u), R = [R; tileRows(n-1, -1, ps)]; end %#ok<AGROW>
        end
      % the phase counted from zero, 0 the notIndexed as in Python
      case 'trueebsd_undistort'
        R = [res.pos.x(:) res.pos.y(:) res.phaseId(:)-1 res.rotations.a(:) res.rotations.b(:) res.rotations.c(:) res.rotations.d(:)];
      % a smoothed map comes back on its grid, in the grid's order
      otherwise, R = [res.pos.x(:) res.pos.y(:) res.rotations.a(:) res.rotations.b(:) res.rotations.c(:) res.rotations.d(:)];
    end
  end

end


function R = tileRows(n, m, ps)
k = numel(ps.peak);
R = [repmat([n m], k, 1) ps.pos.x(:) ps.pos.y(:) ps.u.x(:) ps.u.y(:) ps.peak(:)];
end


% the images, hops and matched grid of the EBSDTrueEbsd page
function page = trueEbsdPage
plottingConvention.default('y↓→x');
ebsd = mtexdata('trueEbsdWCCoSmall','silent');
img = ebsd.opt.trueEbsdImgs;
page.imgList = [mapImage(ebsd.bc, ebsd, 'name','bcImg'), ...
  mapImage(img.fsdB3, 'dxy',img.pixSzImg, 'name','fsdB3'), ...
  mapImage(img.fsdT3, 'dxy',img.pixSzImg, 'name','fsdT3'), ...
  mapImage(img.fsdT1, 'dxy',img.pixSzImg, 'name','fsdT1'), ...
  mapImage(img.fsdT10,'dxy',img.pixSzImg, 'name','fsdT10')];
page.imgList(2:end) = rescale(imboxfilt(page.imgList(2:end),3));
page.hops = @() [spatialTransformShift + spatialTransformDrift, spatialTransformId, spatialTransformShift, spatialTransformTilt];
job = trueEbsd(page.imgList, page.hops());
job.pixelSizeMatch;
page.resized = job.resizedList;
end


% a measured map with its grains, indexed pixels only
function ebsd = prepare(name)
ebsd = mtexdata(name, 'silent');
[~, ebsd] = calcGrains(ebsd, 'angle', 10*degree, 'minPixel', 5);
ebsd = ebsd('indexed');
end


% a map as bench/calcgrains.py exports it, on its grid; phase 1 of CSList is notIndexed
function ebsd = loadNeutral(file)
fid = fopen(file, 'r'); A = fread(fid, [7 inf], 'double', 0, 'ieee-le'); fclose(fid);
lines = splitlines(strtrim(fileread(strrep(file, '.bin', '.phases'))));
CSList = {'notIndexed'};
for k = 1:numel(lines)
  f = strsplit(lines{k}, sprintf('\t'), 'CollapseDelimiters', false);
  al = {};
  if numel(f) > 4 && ~isempty(f{5}), al = strsplit(f{5}, ','); end
  CSList{end+1} = crystalSymmetry(f{2}, str2num(f{3}), str2num(f{4})*degree, al{:}, 'mineral', f{1}); %#ok<ST2NM,AGROW>
end
pos = vector3d(A(1,:).', A(2,:).', zeros(size(A,2),1));
rot = rotation(quaternion(A(4,:).', A(5,:).', A(6,:).', A(7,:).'));
ebsd = gridify(EBSD(pos, rot, A(3,:).' + 1, CSList, struct()));
end
