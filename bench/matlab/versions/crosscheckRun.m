function crosscheckRun(out, T, ops, ~, K, always, reps)
% crosscheckRun of tools/graph for the comparison of MTEX versions (bench/mtex_versions.py), first
% on the path: each operation timed best of five as there, but one a version cannot run is
% reported and left out instead of ending the run, and no profile is written

if nargin < 6, always = {}; end
if nargin < 7, reps = containers.Map; end
for k = 1:size(T,1)
  op = T{k,1};
  if ~any(strcmp(op, ops)) && ~any(strcmp(op, always)), continue; end
  f = T{k,2};
  n = 5; if isKey(reps, op), n = reps(op); end
  try
    if n > 1, f(); end
    t = inf;
    for rep = 1:n, tic; res = f(); t = min(t, toc); end
  catch err
    fprintf('%-20s failed: %s\n', op, err.message);
    continue
  end
  fid = fopen(fullfile(out,['matlab_' op '.txt']),'w'); fprintf(fid,'%.9g\n',t); fclose(fid);
  try
    R = double(T{k,3}(res)); R = R(1:min(K,size(R,1)),:).';
    fid = fopen(fullfile(out,['matlab_' op '.bin']),'w'); fwrite(fid,R(:),'double'); fclose(fid);
  catch
  end
  fprintf('%-20s %9.1f ms\n', op, 1e3*t);
end

end
