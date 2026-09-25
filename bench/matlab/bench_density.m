% one case of bench/density.py in MATLAB: best of three, one profiled run, the values
function bench_density(out, caseName, dom, hw)

k = 3 + strcmp(dom,'SO3');
X = reshape(fread(fopen(fullfile(out,[caseName '.bin'])),'double'),k,[]).';
A = reshape(fread(fopen(fullfile(out,[caseName '_at.bin'])),'double'),k,[]).';
fclose('all');
if strcmp(dom,'SO3')
  cs = crystalSymmetry('m-3m');
  x = orientation(quaternion(X(:,1),X(:,2),X(:,3),X(:,4)),cs);
  at = orientation(quaternion(A(:,1),A(:,2),A(:,3),A(:,4)),cs);
else
  x = vector3d(X(:,1),X(:,2),X(:,3),'antipodal');
  at = vector3d(A(:,1),A(:,2),A(:,3),'antipodal');
end
f = @() calcDensity(x,'halfwidth',hw*degree);

f();
t = inf;
for r = 1:3
  tic; f(); t = min(t,toc);
end
profile clear; profile on;
F = f();
profile off;
profileDump(fullfile(out,['profile_matlab_' caseName '.json']),'density estimation','matlab',caseName,t);

v = F.eval(at);
fid = fopen(fullfile(out,['matlab_' caseName '_values.bin']),'w'); fwrite(fid,double(v(:)),'double'); fclose(fid);
fid = fopen(fullfile(out,['matlab_' caseName '.txt']),'w'); fprintf(fid,'%.9g\n',t); fclose(fid);
fprintf('%s: %.1f ms\n',caseName,1e3*t);

end
