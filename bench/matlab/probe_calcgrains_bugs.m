% the calcGrains defects of the crosscheck (docs/research/benchmark-calcgrains.md), each on
% MTEX's own sample data, for the issue report: the loaded map against the same measurements
% rebuilt as a list and gridified

ebsd = mtexdata('forsterite','silent');
list = EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId, ebsd.CSList, struct()).gridify;
fprintf('loaded %s, list gridified %s\n', mat2str(size(ebsd)), mat2str(size(list)));

% 1. the orientation stored on a notIndexed measurement changes the grains
for m = {ebsd, list}
  e = m{1};
  e1 = e; e1.rotations(~e1.isIndexed) = rotation.nan;
  e2 = e; e2.rotations(~e2.isIndexed) = rotation.id;
  fprintf('1: grains with NaN on notIndexed %d, with the identity %d\n', length(calcGrains(e1)), length(calcGrains(e2)));
end

% 2. neighbouring Enstatite pixels far below the threshold end in different grains
xy = [6100 1100; 6100 1150; 6150 1150; 6200 1150];
for m = {ebsd, list}
  [~, e3] = calcGrains(m{1}, 'angle', 10*degree);
  id = e3.findByLocation(xy);
  o = e3(id).orientations;
  fprintf('2: grainIds %s, angles to the next (deg) %s\n', mat2str(e3.grainId(id)'), mat2str(round(angle(o(1:end-1), o(2:end))'/degree, 3)));
end

% 3. calcGrains fails on a map built from a list and gridified
ebsd = mtexdata('csl','silent');
sub = ebsd(ebsd.x < 60 & ebsd.y < 60);
g = EBSD(sub.pos, sub.rotations, sub.phaseId, sub.CSList, struct()).gridify;
try
  grains = calcGrains(g, 'angle', 10*degree);
  fprintf('3: %d grains\n', length(grains));
catch err
  fprintf('3: failed: %s\n', err.message);
  for s = err.stack(1:min(6, end))', fprintf('   %s line %d\n', s.name, s.line); end
end
