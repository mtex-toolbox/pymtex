function coverageMatlab(fname, tests)
% which MATLAB functions each check_* runs, for build.py: {test id: [[file, 0, name], ...]}
%
%   coverageMatlab('tools/graph/out/coverage_matlab_core.json', {'tests/core/check_Miller', ...})
%
% Each test runs under the profiler in its own try/catch; a failing test keeps what it ran.

res = containers.Map;
for k = 1:numel(tests)
  [d, name] = fileparts(tests{k});
  close all; rng(0);
  profile clear; profile on;
  try
    run(fullfile(mtex_path, d, [name '.m']));
  catch err
    fprintf('%s failed: %s\n', tests{k}, err.message);
  end
  profile off;
  p = profile('info');
  res(tests{k}) = arrayfun(@(t) {t.FileName, 0, t.FunctionName}, p.FunctionTable, 'UniformOutput', false);
  fprintf('%-50s %5d functions\n', tests{k}, numel(p.FunctionTable));
end
fid = fopen(fname,'w'); fwrite(fid, jsonencode(res)); fclose(fid);

end
