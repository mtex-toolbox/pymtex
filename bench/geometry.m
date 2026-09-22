% the MATLAB twin of bench/geometry.py; run from ../frameSymmetry through the bridge,
% see docs/research/benchmark-geometry.md
function geometry(N)
if nargin < 1, N = 1000000; end
cs = crystalFrame('m-3m'); hx = crystalFrame('6/mmm');
rng(0);
r = rotation.rand(N); s = rotation.rand(N);
v = vector3d.rand(N);
ori = orientation.rand(N, cs); ori2 = orientation.rand(N, cs);
ref = ori(1);
mori = orientation.rand(N, cs, hx); mori2 = orientation.rand(N, cs, hx);
mref = mori(1);
m = Miller(vector3d.rand(N), cs);
toRef = @(p, q) mean(abs(dot(p, q, 'noSymmetry')));
rows = {
  'r * s', @() r .* s, @(q) mean(abs(q.a));
  'r * v', @() r .* v, @(w) mean(abs(w.z));
  'dot(ori, ori2)', @() dot(ori, ori2), @mean;
  'dot(ori, ref)', @() dot(ori, ref), @mean;
  'dot(mori, mori2)', @() dot(mori, mori2), @mean;
  'dot(mori, mref)', @() dot(mori, mref), @mean;
  'project2FR(m)', @() project2FundamentalRegion(m), @(w) mean(w.z);
  'project2FR(ori, ref)', @() project2FundamentalRegion(ori, ref), @(p) toRef(p, ref);
  'project2FR(ori, ori2)', @() project2FundamentalRegion(ori, ori2), @(p) toRef(p, ori2);
  'project2FR(mori, mref)', @() project2FundamentalRegion(mori, mref), @(p) toRef(p, mref);
  'project2FR(mori, mori2)', @() project2FundamentalRegion(mori, mori2), @(p) toRef(p, mori2);
};
fprintf('MATLAB %s, N = %d, m-3m and m-3m -> 6/mmm, best of 5, threads = %d\n', version('-release'), N, maxNumCompThreads);
for k = 1:size(rows, 1)
  f = rows{k, 2}; out = f();
  t = inf;
  for rep = 1:5, tic; f(); t = min(t, toc); end
  fprintf('%-26s %9.1f ms %8.1f ns/elem   check %.4f\n', rows{k, 1}, 1e3 * t, 1e9 * t / N, rows{k, 3}(out));
end
end
