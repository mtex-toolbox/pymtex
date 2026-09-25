% the operations of bench/geometry_ops.py in MATLAB
function bench_geometry_ops(out, ops, K, NQ, NR)

load = @(name, k) reshape(fread(fopen(fullfile(out,[name '.bin'])),'double'),k,[]).';
A = load('A',4); B = load('B',4); C = load('C',4); D = load('D',4); V = load('V',3); W = load('W',3);
fclose('all');
q = @(X) quaternion(X(:,1),X(:,2),X(:,3),X(:,4));
cs = crystalSymmetry('m-3m'); hx = crystalSymmetry('6/mmm');
r = rotation(q(A)); s = rotation(q(B)); v = vector3d(V(:,1),V(:,2),V(:,3));
ori = orientation(q(A),cs); ori2 = orientation(q(B),cs);
mori = orientation(q(C),cs,hx); mori2 = orientation(q(D),cs,hx);
m = Miller(vector3d(W(:,1),W(:,2),W(:,3)),cs);
ref = ori(1); mref = mori(1);

benchGrid = [];

quat = @(o) [o.a(:) o.b(:) o.c(:) o.d(:)];
xyz = @(o) [o.x(:) o.y(:) o.z(:)];
col = @(o) o(:);
T = {
  'r_s', @() r .* s, quat;
  'r_v', @() r .* v, xyz;
  'dot_ori_ori2', @() dot(ori, ori2), col;
  'dot_ori_ref', @() dot(ori, ref), col;
  'dot_mori_mori2', @() dot(mori, mori2), col;
  'dot_mori_mref', @() dot(mori, mref), col;
  'p2fr_m', @() project2FundamentalRegion(m), xyz;
  'p2fr_ori_ref', @() project2FundamentalRegion(ori, ref), quat;
  'p2fr_ori_ori2', @() project2FundamentalRegion(ori, ori2), quat;
  'p2fr_mori_mref', @() project2FundamentalRegion(mori, mref), quat;
  'p2fr_mori_mori2', @() project2FundamentalRegion(mori, mori2), quat;
  'grid_build', @() buildGrid(cs), @(g) length(g);
  'grid_nearest', @() nearest(ori(1:NQ)), col;
  'grid_radius', @() radius(ori(1:NR)), @(I) full(sum(I ~= 0, 1 + (size(I,1) == NR))).';
};
names = {'rotation products','symmetry-reduced dot','fundamental region projection','orientation grid search'};
algo = containers.Map(T(:,1), names([1 1 2 2 2 2 3 3 3 3 3 4 4 4]));
crosscheckRun(out, T, ops, algo, K, {'grid_build'});
benchGrid = [];

  function g = buildGrid(cs)
    g = equispacedSO3Grid(cs,'resolution',2.5*degree);
    benchGrid = g;
  end

  % find returns the cosine of half the distance, compared as the angle
  function d = nearest(o)
    [~, d] = find(benchGrid, o);
    d = 2 * acos(min(d, 1));
  end

  function I = radius(o)
    I = find(benchGrid, o, 7.5*degree);
  end

end
