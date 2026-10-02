function crosscheckRun(out, T, ops, algo, K, always, reps)
% the MATLAB half of tools/graph/crosscheck.py: each operation of T = {op, call, rows; ...} asked
% for in ops (or in always) timed best of five, profiled, its first K result rows written; reps
% maps an op to fewer runs, a single one without a warm up; cores is the CPU time of the process
% over the timed runs, every thread's counted, over their wall time

if nargin < 6, always = {}; end
if nargin < 7, reps = containers.Map; end
for k = 1:size(T,1)
  op = T{k,1};
  if ~any(strcmp(op, ops)) && ~any(strcmp(op, always)), continue; end
  f = T{k,2};
  n = 5; if isKey(reps, op), n = reps(op); end
  if n > 1, f(); end
  t = inf; c = cputime; t0 = tic;
  for rep = 1:n, tic; res = f(); t = min(t, toc); end
  cores = (cputime - c) / toc(t0);
  profile clear; profile on; f(); profile off;
  profileDump(fullfile(out,['profile_matlab_' op '.json']), algo(op), 'matlab', op, t, cores);
  fid = fopen(fullfile(out,['matlab_' op '.txt']),'w'); fprintf(fid,'%.9g\n',t); fclose(fid);
  R = double(T{k,3}(res)); R = R(1:min(K,size(R,1)),:).';
  fid = fopen(fullfile(out,['matlab_' op '.bin']),'w'); fwrite(fid,R(:),'double'); fclose(fid);
  fprintf('%-20s %9.1f ms %5.1f cores\n', op, 1e3*t, cores);
end

end
