% the operations of bench/tensors_ops.py in MATLAB
function bench_tensors_ops(out, ops, K)

ld = @(name, k) reshape(fread(fopen(fullfile(out,[name '.bin'])),'double'),k,[]).';
O = ld('O',4); V = ld('V',3);
fclose('all');
ol = crystalSymmetry('mmm', [4.7646 10.2296 5.9942], 'mineral', 'Olivine');
C = stiffnessTensor([320.5 68.15 71.6 0 0 0; 68.15 196.5 76.8 0 0 0; 71.6 76.8 233.5 0 0 0; ...
  0 0 0 64 0 0; 0 0 0 0 77 0; 0 0 0 0 0 78.7], ol, 'density', 3.355);
q = quaternion(O(:,1),O(:,2),O(:,3),O(:,4));
v = vector3d(V(:,1),V(:,2),V(:,3));
ori = orientation(q, ol);
odf = unimodalODF(orientation.byEuler(10*degree, 20*degree, 30*degree, ol), 'halfwidth', 15*degree);
cub = crystalSymmetry('m-3m');
sS = slipSystem.fcc(cub).symmetrise;
epsilon = strainTensor(diag([1 -0.5 -0.5]));
oriT = orientation(q(1:1e4), cub);
R = [];

coords = @(X) reshape(double(X.M), 81, []).';
col = @(x) double(x(:));
T = {
  'rotate', @() rotated(), coords;
  'inv', @() inverted(), coords;
  'average_ori', @() calcTensor(ori, C, 'Hill'), coords;
  'average_odf', @() calcTensor(odf, C), coords;
  'velocity', @() speeds(), @(x) x;
  'youngs', @() C.YoungsModulus(v), col;
  'taylor', @() calcTaylor(inv(oriT) * epsilon, sS), col;
  'taylor_fun', @() calcTaylor(epsilon, sS), @(F) col(F.eval(oriT));
};
names = {'tensor rotation','tensor average','elastic wave velocities','Taylor model'};
algo = containers.Map(T(:,1), names([1 1 2 2 3 3 4 4]));
crosscheckRun(out, T, ops, algo, K, {'rotate'});

  function S = rotated()
    R = ori * C; S = R;
  end
  % nested, since an anonymous function would keep the empty R it was made with
  function S = inverted()
    S = inv(R);
  end
  function x = speeds()
    [vp, vs1, vs2] = C.velocity(v);
    x = [vp(:) vs1(:) vs2(:)];
  end

end
