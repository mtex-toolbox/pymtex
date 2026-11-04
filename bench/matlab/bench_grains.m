function bench_grains(d, name, angle)
% the MATLAB side of bench/grains/run.py: both tracks and the three boundaries of one instance, as
% <d>/res/<port>_<name>_<track>-<boundary>.px.bin (x, y, grainId), .gb.bin (x1, y1, x2, y2, left,
% right) with grainId(:,1) on the left of F(:,1) -> F(:,2), and .txt; port matlab is calcGrains as
% it is, matlabNoStraighten calcGrains(..., 'straighten', false), the boundary pymtex draws
%
%   bench_grains('/path/bench/out/grains/T0', 'T0_01_voronoi', 5)

ebsd = loadMap(d, name);
res = fullfile(d, 'res');
if ~exist(res, 'dir'), mkdir(res); end
for port = ["matlab", "matlabNoStraighten"]
  opt = {'straighten', port == "matlab"};
  for track = ["map", "labels"]
    tic;
    if track == "map"
      [grains, e] = calcGrains(ebsd, 'angle', angle*degree, opt{:});
    else
      [grains, e] = calcGrains(ebsd, 'grainId', opt{:});
    end
    t = toc;
    for s = ["raw", "smooth", "noSimplify"]
      tic;
      switch s
        case "raw", g = grains;
        case "smooth", g = smoothBoundary(grains);
        case "noSimplify", g = smoothBoundary(grains, 1, 'noSimplify');
      end
      writeResult(fullfile(res, sprintf('%s_%s_%s-%s', port, name, track, s)), t + toc, g, e);
    end
  end
end
fprintf('%s: %d grains\n', name, length(grains));
end


function writeResult(stem, t, g, e)
x = e.pos.x(:); y = e.pos.y(:); id = double(e.grainId(:));
ok = isfinite(x) & isfinite(y);
fid = fopen([stem '.px.bin'], 'w'); fwrite(fid, [x(ok) y(ok) id(ok)].', 'double', 0, 'ieee-le'); fclose(fid);
gB = g.boundary;
V = [gB.allV.x(:) gB.allV.y(:)];
fid = fopen([stem '.gb.bin'], 'w');
fwrite(fid, [V(gB.F(:, 1), :) V(gB.F(:, 2), :) double(gB.grainId)].', 'double', 0, 'ieee-le');
fclose(fid);
fid = fopen([stem '.txt'], 'w'); fprintf(fid, 'time %.10g\ngrains %d\n', t, length(g)); fclose(fid);
end


% the neutral map of the instance on its grid, the true labels as ebsd.grainId; phase 1 of CSList is notIndexed
function ebsd = loadMap(d, name)
fid = fopen(fullfile(d, [name '.bin']), 'r');
A = fread(fid, [7 inf], 'double', 0, 'ieee-le');
fclose(fid);
fid = fopen(fullfile(d, [name '.labels.bin']), 'r');
lab = fread(fid, inf, 'double', 0, 'ieee-le');
fclose(fid);
f = strsplit(strtrim(fileread(fullfile(d, [name '.phases']))), sprintf('\t'), 'CollapseDelimiters', false);
CSList = {'notIndexed', crystalSymmetry(f{2}, str2num(f{3}), str2num(f{4})*degree, 'mineral', f{1})}; %#ok<ST2NM>
pos = vector3d(A(1, :).', A(2, :).', zeros(size(A, 2), 1));
rot = rotation(quaternion(A(4, :).', A(5, :).', A(6, :).', A(7, :).'));
ebsd = EBSD(pos, rot, A(3, :).' + 1, CSList, struct('grainId', lab), 'phaseMap', [0 1]);
ebsd = ebsd.gridify;
end
