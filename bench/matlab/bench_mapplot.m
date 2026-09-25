function bench_mapplot(fname, outFile)
% the inverse pole figure colouring and the plot of one EBSD map in MTEX, for docs/research/benchmark-mapplot.md
ebsd = EBSD.load(fname, 'silent');
s.n = length(ebsd);
ind = ebsd('indexed');
% the key on the orientations of the largest phase
ids = ind.phaseId; big = mode(ids);
ori = ebsd(ebsd.phaseId == big).orientations;
key = ipfHSVKey(ori.CS);
t = tic; c = key.orientation2color(ori); s.key = toc(t);
for k = 1:2
  t = tic; c = key.orientation2color(ori); s.key = min(s.key, toc(t));
end
% the key through its precomputed grid, the grid built once
keyGrid = ipfHSVKey(ori.CS);
t = tic; keyGrid.precompute; s.keyPrecompute = toc(t);
c2 = keyGrid.orientation2color(ori); s.keyGrid = inf;
for k = 1:3
  t = tic; c2 = keyGrid.orientation2color(ori); s.keyGrid = min(s.keyGrid, toc(t));
end
s.keyGridDeviation = max(abs(c2(:) - c(:)));
% the map of the largest phase drawn and written to a png
f = figure('Visible', 'off');
t = tic; plot(ebsd(ebsd.phaseId == big), c); drawnow; s.plot = toc(t);
t = tic; print(f, '-dpng', '-r150', [outFile '.png']); s.save = toc(t);
close(f);
% the same map coloured by the plot itself, MATLAB's default inverse pole figure key
sub = ebsd(ebsd.phaseId == big);
f = figure('Visible', 'off');
t = tic; plot(sub, sub.orientations); drawnow; s.plotIPF = toc(t);
close(f);
fid = fopen(outFile, 'w'); fwrite(fid, jsonencode(s)); fclose(fid);
end
