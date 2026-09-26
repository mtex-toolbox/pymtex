function probe_pseudosym()
% the numbers of doc/EBSDAnalysis/EBSDPseudoSymmetry.m, for mtex/maps/pseudosym.py
ebsd = mtexdata('forsterite','silent');
ebsd = ebsd('indexed');
[grains,ebsd] = calcGrains(ebsd,'angle',10*degree,'minPixel',5);
cs = ebsd('Fo').CS;
psSym = orientation.byAxisAngle(Miller(1,0,0,cs,'uvw'),60*degree);
gB = grains.boundary('Fo','Fo');
gB = gB(any(angle(gB.misorientation,psSym) < 2*degree,2));
compId = gB.componentId;
len = accumarray(compId,gB.segLength);
xmin = accumarray(compId,gB.midPoint.x,[],@min); xmax = accumarray(compId,gB.midPoint.x,[],@max);
ymin = accumarray(compId,gB.midPoint.y,[],@min); ymax = accumarray(compId,gB.midPoint.y,[],@max);
tort = len ./ sqrt((xmax-xmin).^2 + (ymax-ymin).^2);
fprintf('grains %d, pseudo boundary %d segments in %d components, non-finite %d, above 1.5 %d\n', length(grains), length(gB), max(compId), nnz(~isfinite(tort)), nnz(tort > 1.5));
[ebsdC,grainsC,numChanged] = cleanUpPseudoSym(ebsd,grains,psSym);
fprintf('changed %d, grains after %d\n', numChanged, length(grainsC));
end
