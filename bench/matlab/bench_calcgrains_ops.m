% the operations of bench/calcgrains_ops.py in MATLAB
function bench_calcgrains_ops(out, ops, K)

maps = struct;
names = {'small','peridotite_2.1','peridotite_6.1','peridotite_16.1','forsterite','forsterite_5','cracks', ...
  'peridotite_15','peridotite_pocket10','peridotite_pocket10_crack15','peridotite_delaunay'};
args = {{'angle',10*degree,'minPixel',5}, {'alpha',2.1,'minPixel',5}, {'alpha',6.1,'minPixel',5}, ...
  {'alpha',16.1,'minPixel',5}, {}, {'minPixel',5}, {'angle',5*degree}, {'angle',15*degree,'minPixel',5}, ...
  {'angle',15*degree,'minPixel',5,'alphaPocket',10}, {'angle',15*degree,'minPixel',5,'alphaPocket',10,'alphaCrack',15}, ...
  {'angle',15*degree,'minPixel',5,'delaunay'}};
src = {'small','peridotite','peridotite','peridotite','forsterite','forsterite','T1_clean_cracks', ...
  'peridotite','peridotite','peridotite','peridotite'};
ops = cellstr(ops);
for k = 1:numel(names)
  if any(startsWith(ops, names{k})) && ~isfield(maps, fieldName(src{k})), maps.(fieldName(src{k})) = load(src{k}); end
end
if any(strcmp(ops, 'peridotite_kinds')) && ~isfield(maps, 'peridotite'), maps.peridotite = load('peridotite'); end
allOps = [names, strcat(names, '_bisectors'), {'peridotite_kinds'}];
T = cell(numel(allOps), 3);
for k = 1:2*numel(names)
  i = mod(k-1, numel(names)) + 1;
  T(k,:) = {allOps{k}, @() calcGrains(maps.(fieldName(src{i})), args{i}{:}, 'straighten', k <= numel(names)), @rows};
end
% the kind of every pixel of peridotite at 15 degree
T(end,:) = {'peridotite_kinds', @() classifyNotIndexed(maps.peridotite, 'angle', 15*degree), @(k) kindRows(k, maps.peridotite)};
algos = [repmat({'boundary straightening'}, 1, numel(names)), repmat({'grain segmentation'}, 1, numel(names)), {'notIndexed kinds'}];
crosscheckRun(out, T, ops, containers.Map(allOps, algos), K, {}, containers.Map(allOps, num2cell(3*ones(1, numel(allOps)))));

end


function f = fieldName(name)
f = matlab.lang.makeValidName(name);
end


function ebsd = load(name)
if startsWith(name, 'T1_'), ebsd = loadT1(name); else, ebsd = mtexdata(name, 'silent'); end
end


% the grain count, then the segments of both boundaries between two grains
function R = rows(g)
R = [length(g) nnz(g.isIndexed) 0 0];
for gB = {g.boundary, g.innerBoundary}
  V = [gB{1}.allV.x(:) gB{1}.allV.y(:)];
  ok = all(gB{1}.grainId > 0, 2);
  R = [R; V(gB{1}.F(ok,1),:) V(gB{1}.F(ok,2),:)]; %#ok<AGROW>
end
end


% the position of every pixel and the code of its kind, 1 pore to 6 indexed
function R = kindRows(kinds, ebsd)
R = [ebsd.x(:) ebsd.y(:) double(kinds(:)) zeros(numel(kinds), 1)];
end


% an instance of the grain benchmark on its grid, as bench_grains.m reads it
function ebsd = loadT1(name)
d = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'out', 'grains', 'T1');
fid = fopen(fullfile(d, [name '.bin']), 'r'); A = fread(fid, [7 inf], 'double', 0, 'ieee-le'); fclose(fid);
lines = splitlines(strtrim(fileread(fullfile(d, [name '.phases']))));
CSList = {'notIndexed'};
for k = 1:numel(lines)
  f = strsplit(lines{k}, sprintf('\t'), 'CollapseDelimiters', false);
  al = {};
  if numel(f) > 4 && ~isempty(f{5}), al = strsplit(f{5}, ','); end
  CSList{end+1} = crystalSymmetry(f{2}, str2num(f{3}), str2num(f{4})*degree, al{:}, 'mineral', f{1}); %#ok<ST2NM,AGROW>
end
pos = vector3d(A(1,:).', A(2,:).', zeros(size(A,2),1));
rot = rotation(quaternion(A(4,:).', A(5,:).', A(6,:).', A(7,:).'));
ebsd = EBSD(pos, rot, A(3,:).' + 1, CSList, struct(), 'phaseMap', 0:numel(CSList)-1);
ebsd = ebsd.gridify;
end
