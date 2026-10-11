% the operations of bench/analysis_ops.py in MATLAB
function bench_analysis_ops(out, ops, K)

ld = @(name, k) reshape(fread(fopen(fullfile(out,[name '.bin'])),'double'),k,[]).';
O = ld('O',4); V = ld('V',3); C = ld('C',4); G = ld('G',3);
R = ld('R',4); Rw = ld('Rw',1); X = ld('X',3); D = ld('D',3); Q = ld('Q',4); HS = ld('HS',3); HO = ld('HO',4);
fclose('all');
q = @(X) quaternion(X(:,1),X(:,2),X(:,3),X(:,4));
cub = crystalSymmetry('m-3m');
ori = orientation(q(O), cub);
sS = slipSystem.fcc(cub).symmetrise;
ol = crystalSymmetry('mmm', [4.756 10.207 5.980], 'mineral', 'Forsterite');
rI = 0.86 * refractiveIndexTensor(diag([1.670 1.635 1.651]), ol) + 0.14 * refractiveIndexTensor(diag([1.879 1.827 1.869]), ol);
v = vector3d(V(:,1),V(:,2),V(:,3));
g = vector3d(G(:,1),G(:,2),G(:,3));
% 300 random centres in m-3m, a 10 degree kernel each
odf = SO3FunRBF(orientation(q(R), cub), SO3DeLaValleePoussinKernel('halfwidth',10*degree), Rw);
% a unimodal ODF of 20 degree without symmetry
unimodal = SO3FunRBF(orientation(rotation.byEuler(10*degree,20*degree,30*degree), crystalSymmetry('1')), ...
  SO3DeLaValleePoussinKernel('halfwidth',20*degree), 1);
d = vector3d(D(:,1),D(:,2),D(:,3));
qo = orientation(q(Q), crystalSymmetry('432'));
hs = vector3d(HS(:,1),HS(:,2),HS(:,3));
ho = orientation(q(HO), cub);
% CLASSIX as calcCluster calls it
copts = struct('merge_scale', 1.5, 'merge_tiny_groups', 1, 'use_mex', 0);
dc06 = []; tw = []; gB = [];
% dc06 as the GND page loads it
% the files of one data directory for every version; a version without what a setup needs
% leaves the ops that use it out
data = fullfile(getenv('HOME'), '.mtex', 'data');
if any(strcmp(ops,'gnd')), try, dc06 = EBSD.load(fullfile(data,'EBSD','DC06_2uniax.ang'),'setting',2); catch, end, end
if any(strcmp(ops,'gbnd'))
  try
    tw = mtexdata('twins','silent');
    grains = calcGrains(tw, 'angle', 5*degree, 'minPixel', 3);
    gB = grains.boundary('indexed');
  catch
  end
end

col = @(x) double(x(:));
quat = @(o) [o.a(:) o.b(:) o.c(:) o.d(:)];
T = {
  'gnd', @() gnd(), @(x) x;
  'gbnd', @() calcGBND(gB, tw, 'halfwidth', 5*degree, 'nonneg'), @(f) col(real(f.eval(g)));
  'schmid', @() sS.SchmidFactor(inv(ori) * xvector), @(m) double(reshape(m, [], 24));
  'birefringence', @() rI.birefringence(v), col;
  'components', @() components(), @(x) x;
  'odf_max', @() odfMax(), @(x) x;
  'cluster', @() clusterCentres(), quat;
  'classix', @() classix(X, 0.1, 2000, copts), @(c) [cell2mat(arrayfun(@(k) mean(X(c==k,:),1), (1:max(c)).', 'UniformOutput', false)) accumarray(c(:),1)];
  'cluster_directions', @() clusterOf(d, 'radius', 0.1, 'minPoints', 1000), @(r) [r{2}.x(:) r{2}.y(:) r{2}.z(:) accumarray(r{1}(:),1)];
  'cluster_orientations', @() clusterOf(qo, 'method', 'classix', 'radius', 0.1, 'minPoints', 200), @(r) [quat(r{2}) accumarray(r{1}(:),1)];
  'optimal_s2', @() optimalSample(S2FunHarmonic([sqrt(4*pi); 0; 0.6*sqrt(4*pi/3); 0]), 300, 'bandwidth', 16), @(x) [x.x(:) x.y(:) x.z(:)];
  'optimal_so3', @() optimalSample(unimodal, 200, 'bandwidth', 16), quat;
  'halfwidth_s2_ucv', @() selectHalfwidth(hs, 'UCV'), col;
  'halfwidth_s2_cons', @() selectHalfwidth(hs, 'conservative'), col;
  'halfwidth_so3_ucv', @() selectHalfwidth(ho, 'UCV'), col;
  'halfwidth_so3_cons', @() selectHalfwidth(ho, 'conservative'), col;
};
names = {'dislocation density','grain boundary normal distribution','Schmid factor','birefringence','ODF components','ODF components', ...
  'clustering','clustering','clustering','clustering','discrete sampling','discrete sampling', ...
  'halfwidth selection','halfwidth selection','halfwidth selection','halfwidth selection'};
algo = containers.Map(T(:,1), names);
once = T(~ismember(T(:,1), {'schmid','birefringence'}), 1);
crosscheckRun(out, T, ops, algo, K, {}, containers.Map(once, num2cell(ones(size(once)))));

  function R = gnd()
    dS = dislocationSystem.bcc(dc06.CS);
    dS(dS.isEdge).u = 1;
    dS(dS.isScrew).u = 1 - 0.3;
    x = calcGND(dc06, dS);
    R = [dc06.pos.x(:) dc06.pos.y(:) x(:)];
  end
  function R = components()
    [modes, w] = calcComponents(odf, 'silent');
    R = [quat(modes) w(:)];
  end
  function R = odfMax()
    [v, o] = max(odf);
    R = [v quat(o)];
  end
  function r = clusterOf(x, varargin)
    [cId, c] = calcCluster(x, varargin{:});
    r = {cId, c};
  end
  function c = clusterCentres()
    [~, c] = calcCluster(orientation(q(C), crystalSymmetry('432')), 'numCluster', 4, 'method', 'hierarchical');
  end

end
