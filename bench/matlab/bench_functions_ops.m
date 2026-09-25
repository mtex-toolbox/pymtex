% the operations of bench/functions_ops.py in MATLAB
function bench_functions_ops(out, ops, K)

ld = @(name, k) reshape(fread(fopen(fullfile(out,[name '.bin'])),'double'),k,[]).';
C = ld('C',4); w = ld('w',1); O = ld('O',4); V = ld('V',3); fO = ld('fO',1); fV = ld('fV',1);
fclose('all');
q = @(X) quaternion(X(:,1),X(:,2),X(:,3),X(:,4));
cs = crystalSymmetry('m-3m'); tri = crystalSymmetry('1');
v = vector3d(V(:,1),V(:,2),V(:,3));
ori = orientation(q(O),cs); oriT = orientation(q(O),tri);
odf = SO3FunRBF(orientation(q(C),cs), SO3DeLaValleePoussinKernel('halfwidth',10*degree), w);
F = []; G = [];
wO = ones(size(fO)) / numel(fO); wV = 4*pi*ones(size(fV)) / numel(fV);

coef = @(H) [real(H.fhat(:)) imag(H.fhat(:))];
col = @(x) real(x(:));
T = {
  'rbf_harmonic', @() harmonic(), coef;
  'rbf_direct', @() odf.eval(ori(1:1e4)), col;
  'so3_eval', @() evalF(ori), col;
  'so3_quadrature', @() SO3FunHarmonic.quadrature(oriT, fO, 'bandwidth', 32, 'weights', wO), coef;
  's2_quadrature', @() quadS2(), coef;
  's2_eval', @() evalG(v), col;
  'so3_interpolate', @() SO3FunHarmonic.interpolate(oriT(1:20000), fO(1:20000), 'bandwidth', 16), @(H) col(H.eval(oriT(1:1e4)));
  's2_interpolate', @() S2FunHarmonic.interpolate(v(1:20000), fV(1:20000), 'bandwidth', 48), @(H) col(H.eval(v(1:1e4)));
};
names = {'harmonic transforms','radial basis functions','least squares approximation'};
algo = containers.Map({'so3_eval','so3_quadrature','s2_quadrature','s2_eval','rbf_direct','rbf_harmonic','so3_interpolate','s2_interpolate'}, ...
  names([1 1 1 1 2 2 3 3]));
crosscheckRun(out, T, ops, algo, K, {'rbf_harmonic','s2_quadrature'});
fprintf('harmonic form against the kernel sum: %.1e\n', max(abs(F.eval(ori(1:1e4)) - odf.eval(ori(1:1e4)))));

  function H = harmonic()
    F = SO3FunHarmonic(odf, 'bandwidth', 64); H = F;
  end
  function x = evalF(o)
    x = F.eval(o);
  end
  function H = quadS2()
    G = S2FunHarmonic.quadrature(v, fV, 'bandwidth', 128, 'weights', wV); H = G;
  end
  function x = evalG(u)
    x = G.eval(u);
  end

end
