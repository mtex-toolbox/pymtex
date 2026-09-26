% the operations of bench/analysis_ops.py in MATLAB
function bench_analysis_ops(out, ops, K)

ld = @(name, k) reshape(fread(fopen(fullfile(out,[name '.bin'])),'double'),k,[]).';
O = ld('O',4); V = ld('V',3); C = ld('C',4); G = ld('G',3);
fclose('all');
q = @(X) quaternion(X(:,1),X(:,2),X(:,3),X(:,4));
cub = crystalSymmetry('m-3m');
ori = orientation(q(O), cub);
sS = slipSystem.fcc(cub).symmetrise;
ol = crystalSymmetry('mmm', [4.756 10.207 5.980], 'mineral', 'Forsterite');
rI = 0.86 * refractiveIndexTensor(diag([1.670 1.635 1.651]), ol) + 0.14 * refractiveIndexTensor(diag([1.879 1.827 1.869]), ol);
v = vector3d(V(:,1),V(:,2),V(:,3));
g = vector3d(G(:,1),G(:,2),G(:,3));
dc06 = []; tw = []; gB = [];
% dc06 as the GND page loads it
if any(strcmp(ops,'gnd')), dc06 = EBSD.load(fullfile(mtexDataPath,'EBSD','DC06_2uniax.ang'),'setting',2); end
if any(strcmp(ops,'gbnd'))
  tw = mtexdata('twins','silent');
  grains = calcGrains(tw, 'angle', 5*degree, 'minPixel', 3);
  gB = grains.boundary('indexed');
end

col = @(x) double(x(:));
quat = @(o) [o.a(:) o.b(:) o.c(:) o.d(:)];
T = {
  'gnd', @() gnd(), @(x) x;
  'gbnd', @() calcGBND(gB, tw, 'halfwidth', 5*degree, 'nonneg'), @(f) col(real(f.eval(g)));
  'schmid', @() sS.SchmidFactor(inv(ori) * xvector), @(m) double(reshape(m, [], 24));
  'birefringence', @() rI.birefringence(v), col;
  'cluster', @() clusterCentres(), quat;
  'optimal_s2', @() optimalSample(S2FunHarmonic([sqrt(4*pi); 0; 0.6*sqrt(4*pi/3); 0]), 300, 'bandwidth', 16), @(x) [x.x(:) x.y(:) x.z(:)];
};
names = {'dislocation density','grain boundary normal distribution','Schmid factor','birefringence','clustering','discrete sampling'};
algo = containers.Map(T(:,1), names);
crosscheckRun(out, T, ops, algo, K, {}, containers.Map({'gnd','gbnd','cluster','optimal_s2'}, {1, 1, 1, 1}));

  function R = gnd()
    dS = dislocationSystem.bcc(dc06.CS);
    dS(dS.isEdge).u = 1;
    dS(dS.isScrew).u = 1 - 0.3;
    x = calcGND(dc06, dS);
    R = [dc06.pos.x(:) dc06.pos.y(:) x(:)];
  end
  function c = clusterCentres()
    [~, c] = calcCluster(orientation(q(C), crystalSymmetry('432')), 'numCluster', 4, 'method', 'hierarchical');
  end

end
