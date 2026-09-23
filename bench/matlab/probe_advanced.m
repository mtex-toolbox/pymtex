% the refined OR of the MaParentGrainReconstructionAdvanced page, as a quaternion
warning('off','all'); rmpath(genpath('/home/hielscher/mtex/master')); warning('on','all');
mtexdata martensite silent
csBCC = ebsd.CSList(2); csFCC = ebsd.CSList(3);
[grains,ebsd] = calcGrains(ebsd,'angle',3*degree,'minPixel',5,'removeQuadruplePoints');
grainPairs = neighbors(grains('Iron bcc'));
pairOri = grains(grainPairs).meanOrientation;
grainPairs(angle(pairOri(:,1),pairOri(:,2)) < 5*degree,:) = [];
fcc2bcc = calcParent2Child(grains(grainPairs).meanOrientation,orientation.KurdjumovSachs(csFCC,csBCC));
fprintf('p2c %s\n', mat2str([fcc2bcc.a fcc2bcc.b fcc2bcc.c fcc2bcc.d],12));
fprintf('grains %d pairs %d\n', length(grains), size(grainPairs,1));
