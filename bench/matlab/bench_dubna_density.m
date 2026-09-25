% one step of bench/dubna_density.py: calcODF, discreteSample or calcDensity, the result kept for the next
function bench_dubna_density(out, step, timeIt, reps)
global benchState

switch step
  case 'calcODF'
    pf = mtexdata('dubna','silent');
    f = @() calcODF(pf,'silent');
    f();
  case 'discreteSample'
    rng(0);
    f = @() benchState.odf.discreteSample(1e7);
  case 'calcDensity'
    f = @() calcDensity(benchState.ori);
end
names = struct('calcODF','pole figure inversion','discreteSample','discrete sampling','calcDensity','density estimation');

if timeIt
  t = inf;
  for r = 1:reps
    tic; res = f(); t = min(t,toc);
  end
  profile clear; profile on;
  f();
  profile off;
  profileDump(fullfile(out,['profile_matlab_' step '.json']),names.(step),'matlab','dubna',t);
  fid = fopen(fullfile(out,['matlab_' step '.txt']),'w'); fprintf(fid,'%.9g\n',t); fclose(fid);
  fprintf('%-16s %6.2f s\n',step,t);
else
  res = f();
end

switch step
  case 'calcODF', benchState = struct('odf',res);
  case 'discreteSample', benchState.ori = res;
  case 'calcDensity'
    A = reshape(fread(fopen(fullfile(out,'at.bin')),'double'),4,[]).'; fclose('all');
    at = orientation(quaternion(A(:,1),A(:,2),A(:,3),A(:,4)),benchState.odf.CS,benchState.odf.SS);
    writeValues(fullfile(out,'matlab_odf_values.bin'),benchState.odf.eval(at));
    writeValues(fullfile(out,'matlab_density_values.bin'),res.eval(at));
    benchState = [];
end

end

function writeValues(fname,v)
fid = fopen(fname,'w'); fwrite(fid,real(double(v(:))),'double'); fclose(fid);
end
