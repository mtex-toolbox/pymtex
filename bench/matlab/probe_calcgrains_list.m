rd = @(n) fread(fopen(['/home/hielscher/mtex/pymtex/bench/out/calcgrains/' n '.bin'],'r'),[7 inf],'double',0,'ieee-le');
C = rd('csl'); fclose('all');
cs = crystalFrame('m-3m','mineral','iron');
mk = @(P) EBSD(vector3d(P(1,:).',P(2,:).',0*P(1,:).'), rotation(quaternion(P(4,:).',P(5,:).',P(6,:).',P(7,:).')), P(3,:).'+1, {'notIndexed',cs}, struct());
sets = {'full', C; 'middle', C(:, C(1,:) >= 100 & C(1,:) < 160 & C(2,:) >= 100 & C(2,:) < 160); 'corner', C(:, C(1,:) < 60 & C(2,:) < 60); 'corner 200', C(:, C(1,:) < 200 & C(2,:) < 200)};
for k = 1:size(sets,1)
  P = sets{k,2};
  try, g = calcGrains(mk(P).gridify,'angle',10*degree); fprintf('%s (%d px): %d grains\n', sets{k,1}, size(P,2), length(g));
  catch err, fprintf('%s (%d px): failed %s\n', sets{k,1}, size(P,2), err.message); end
end
