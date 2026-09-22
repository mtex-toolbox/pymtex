% the MATLAB twin of bench/gridsearch.py; run from ../frameSymmetry through the bridge,
% see docs/research/benchmark-gridsearch.md. find uses the compiled MEX ring search
function gridsearch(nn)
if nargin < 1, nn = 10000; end
res = 2.5*degree; eps = 7.5*degree; nr = 2000;
cs = crystalFrame('m-3m');
rng(0);
tic; S3G = equispacedSO3Grid(cs, 'resolution', res); tbuild = toc;
o = orientation.rand(nn, cs);
find(S3G, o(1:5)); find(S3G, o(1:5), eps);            % warm
tn = inf; for r=1:3, tic; [ind,d] = find(S3G, o); tn = min(tn, toc); end
tr = inf; for r=1:3, tic; I = find(S3G, o(1:nr), eps); tr = min(tr, toc); end
fprintf('MATLAB %s, cubic m-3m %.1f deg grid\n', version('-release'), res/degree);
fprintf('  build            %8.0f ms   (%d points)\n', 1e3*tbuild, length(S3G));
fprintf('  nearest %6d   %8.1f ms   %6.1f us/query\n', nn, 1e3*tn, 1e6*tn/nn);
fprintf('  radius  %6d   %8.1f ms   %6.1f us/query   mean %.1f nb @ %.1f deg\n', nr, 1e3*tr, 1e6*tr/nr, nnz(I)/nr, eps/degree);
end
