% one-pixel grains per phase of the emsland map at 5 degrees, bench/matlab of the
% MartensiteVariants comparison
warning('off','all'); rmpath(genpath('/home/hielscher/mtex/master')); warning('on','all');
mtexdata emsland silent
grains = calcGrains(ebsd,'angle',5*degree);
for ph = {'Aus','Fe'}
  s = grains(ph{1}).numPixel; fprintf('%s %d %d\n', ph{1}, length(s), nnz(s==1));
end
[grains2,ebsd2] = calcGrains(ebsd,'angle',5*degree,'minPixel',2);
fprintf('after %d %d %d %d\n', length(ebsd2('Aus')), length(ebsd2('Fe')), length(grains2('Fe')), length(grains2('Aus')));
