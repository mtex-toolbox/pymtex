function bench_calcgrains(outDir, name, caseName)
% the MATLAB side of bench/calcgrains.py: one map and one case per call, the best of three
% after a warm up, the counts to <outDir>/matlab_<name>_<case>.txt and the partition as
% float64 rows (x, y, grainId) to <outDir>/matlab_<name>_<case>.bin
%
%   bench_calcgrains('/path/bench/out/calcgrains', 'twins', 'a')

% the cases of bench/calcgrains.py: angle in degrees, minPixel, alpha, removeQuadruplePoints
cases.a = {10, 1, 3.1, false};
cases.b = {5, 5, 3.1, false};
cases.c = {10, 1, 0, false};
cases.d = {10, 1, 3.1, true};
c = cases.(caseName);
opts = {'angle', c{1}*degree, 'minPixel', c{2}, 'alpha', c{3}};
if c{4}, opts{end+1} = 'removeQuadruplePoints'; end

ebsd = loadMap(outDir, name);

[~, ~] = calcGrains(ebsd, opts{:});
t = inf;
for k = 1:3
  tic;
  [grains, ebsdOut] = calcGrains(ebsd, opts{:});
  t = min(t, toc);
end

ind = grains.isIndexed;
s = {'time', t; 'grains', length(grains); 'indexed', nnz(ind); 'notIndexed', nnz(~ind); ...
  'boundary', length(grains.boundary); 'inner', length(grains.innerBoundary); ...
  'triplePoints', length(grains.triplePoints); 'largest', max([0; grains.numPixel(ind)]); ...
  'threads', maxNumCompThreads};
fid = fopen(fullfile(outDir, sprintf('matlab_%s_%s.txt', name, caseName)), 'w');
for k = 1:size(s, 1), fprintf(fid, '%s %.10g\n', s{k, 1}, s{k, 2}); end
fclose(fid);

x = ebsdOut.pos.x(:); y = ebsdOut.pos.y(:); g = double(ebsdOut.grainId(:));
ok = isfinite(x) & isfinite(y);
fid = fopen(fullfile(outDir, sprintf('matlab_%s_%s.bin', name, caseName)), 'w');
fwrite(fid, [x(ok) y(ok) g(ok)].', 'double', 0, 'ieee-le');
fclose(fid);
fprintf('%s %s: %.3f s, %d grains\n', name, caseName, t, length(grains));

end


% the map bench/calcgrains.py exported, on its grid; phase 1 of CSList is notIndexed
function ebsd = loadMap(outDir, name)
% gridify of a hexagonal list winds the unit cell clockwise and calcGrains fails in grain2d, so
% the copper map is read by MATLAB's own reader, whose positions are those of the export
if strcmp(name, 'copper')
  ebsd = mtexdata('copper', 'silent');
  return
end
fid = fopen(fullfile(outDir, [name '.bin']), 'r');
A = fread(fid, [7 inf], 'double', 0, 'ieee-le');
fclose(fid);
lines = splitlines(strtrim(fileread(fullfile(outDir, [name '.phases']))));
CSList = {'notIndexed'};
for k = 1:numel(lines)
  f = strsplit(lines{k}, sprintf('\t'), 'CollapseDelimiters', false);
  al = {};
  if numel(f) > 4 && ~isempty(f{5}), al = strsplit(f{5}, ','); end
  CSList{end+1} = crystalSymmetry(f{2}, str2num(f{3}), str2num(f{4})*degree, al{:}, 'mineral', f{1}); %#ok<ST2NM,AGROW>
end
pos = vector3d(A(1, :).', A(2, :).', zeros(size(A, 2), 1));
rot = rotation(quaternion(A(4, :).', A(5, :).', A(6, :).', A(7, :).'));
% The exported phases are indices, not vendor phase numbers. In particular,
% forsterite includes an unused Silicon frame: inferring the phase map from
% the phases present shifts every index and assigns the wrong crystal group.
phaseId = A(3, :).' + 1;
ebsd = EBSD(pos, rot, phaseId, CSList, struct(), 'phaseMap', 0:numel(CSList)-1);
assert(isequal(ebsd.phaseId(:), phaseId));
assert(isequal(ebsd.isIndexed(:), A(3, :).' > 0));
ebsd = ebsd.gridify;
end
