function probe_ebsd(fname, outFile)
% one EBSD file read by MTEX, its numbers written as JSON for the comparison with the port
t = tic;
ebsd = EBSD.load(fname, 'noGrid', 'silent');
s.time = toc(t);
s.n = length(ebsd);
s.phaseMap = reshape(ebsd.phaseMap, 1, []);
cs = ebsd.CSList;
if ~iscell(cs), cs = num2cell(cs); end
s.counts = arrayfun(@(k) nnz(ebsd.phaseId == k), 1:numel(cs));
s.minerals = cellfun(@(c) char(c.mineral), cs, 'UniformOutput', false);
s.pointGroups = cellfun(@(c) pgName(c), cs, 'UniformOutput', false);
s.abc = cellfun(@(c) cellOf(c), cs, 'UniformOutput', false);
ind = ebsd.isIndexed;
s.nIndexed = nnz(ind);
k = find(ind, 1);
s.firstIndexed = k;
if ~isempty(k)
  [a, b, c] = Euler(ebsd.rotations(k), 'Bunge');
  s.firstEuler = [a b c] / degree;
  s.firstPos = [ebsd.pos(k).x, ebsd.pos(k).y];
  M = matrix(ebsd.rotations(ind));
  s.meanMatrix = mean(M, 3);
end
s.extent = [min(ebsd.pos.x) max(ebsd.pos.x) min(ebsd.pos.y) max(ebsd.pos.y)];
fn = fieldnames(ebsd.prop);
s.props = struct();
for i = 1:numel(fn)
  v = ebsd.prop.(fn{i});
  if isnumeric(v), s.props.(lower(fn{i})) = mean(double(v(:)), 'omitnan'); end
end
fid = fopen(outFile, 'w'); fwrite(fid, jsonencode(s)); fclose(fid);
end

function n = pgName(cs)
if isa(cs, 'crystalFrame'), n = char(cs.pointGroup); else, n = ''; end
end

function v = cellOf(cs)
if isa(cs, 'crystalFrame'), v = [cs.abc(:); cs.abg(:) / degree]'; else, v = []; end
end
