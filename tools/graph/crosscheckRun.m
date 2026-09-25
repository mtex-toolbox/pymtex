function crosscheckRun(out, T, ops, algo, K, always)
% the MATLAB half of tools/graph/crosscheck.py: each operation of T = {op, call, rows; ...} asked
% for in ops (or in always) timed best of five, profiled, its first K result rows written

if nargin < 6, always = {}; end
for k = 1:size(T,1)
  op = T{k,1};
  if ~any(strcmp(op, ops)) && ~any(strcmp(op, always)), continue; end
  f = T{k,2}; res = f();
  t = inf;
  for rep = 1:5, tic; f(); t = min(t, toc); end
  profile clear; profile on; f(); profile off;
  profileDump(fullfile(out,['profile_matlab_' op '.json']), algo(op), 'matlab', op, t);
  fid = fopen(fullfile(out,['matlab_' op '.txt']),'w'); fprintf(fid,'%.9g\n',t); fclose(fid);
  R = double(T{k,3}(res)); R = R(1:min(K,size(R,1)),:).';
  fid = fopen(fullfile(out,['matlab_' op '.bin']),'w'); fwrite(fid,R(:),'double'); fclose(fid);
  fprintf('%-20s %9.1f ms\n', op, 1e3*t);
end

end
