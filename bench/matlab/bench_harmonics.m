% the cases of bench/harmonics_user.py in MATLAB's master, one section per tag
function bench_harmonics(tag)

rng(0);
tri = crystalSymmetry('1'); cub = crystalSymmetry('m-3m'); ss1 = specimenSymmetry;

switch tag
  case 'eval'
    for c = {{64,1e6},{128,1e4},{128,1e6}}
      [L,J] = c{1}{:};
      for cs = {tri,cub}
        F = randomODF(L,cs{1},ss1);
        ori = orientation.rand(J,cs{1},ss1);
        report(sprintf('SO3 eval L=%d J=%.0e %s',L,J,cs{1}.pointGroup), timeit(@() F.eval(ori),2));
      end
    end
  case 'adjoint'
    for c = {{64,1e6},{128,1e4},{128,1e6}}
      [L,J] = c{1}{:};
      for cs = {tri,cub}
        ori = orientation.rand(J,cs{1},ss1);
        v = rand(J,1);
        report(sprintf('SO3 quadrature L=%d J=%.0e %s',L,J,cs{1}.pointGroup), ...
          timeit(@() SO3FunHarmonic.quadrature(ori,v,'bandwidth',L),2));
      end
    end
  case 'grid'
    pairs = {{127,tri,ss1,'1/1'},{127,crystalSymmetry('432'),ss1,'432/1'},{128,crystalSymmetry('622'),specimenSymmetry('222'),'622/222'}};
    for c = pairs
      [L,cs,ss,name] = c{1}{:};
      F = randomODF(L,cs,ss);
      g = quadratureSO3Grid(L,cs,ss);
      report(sprintf('SO3 eval Clenshaw-Curtis L=%d %s',L,name), timeit(@() F.eval(g),2));
      v = F.eval(g);
      report(sprintf('SO3 quadrature Clenshaw-Curtis L=%d %s',L,name), timeit(@() SO3FunHarmonic.quadrature(g,v),2));
    end
  case 'sym'
    pairs = {{crystalSymmetry('622'),specimenSymmetry('222'),'622/222'},{crystalSymmetry('432'),specimenSymmetry('mmm'),'432/mmm'},{crystalSymmetry('432'),ss1,'432/1'}};
    for c = pairs
      [cs,ss,name] = c{1}{:};
      fhat = randn(deg2dim(129),1)+1i*randn(deg2dim(129),1);
      report(sprintf('SO3 symmetrise L=128 %s',name), timeit(@() SO3FunHarmonic(fhat,cs,ss),3));
    end
  case 'misc'
    F = randomODF(128,tri,ss1);
    report('SO3 isReal L=128', timeit(@() F.isReal,3));
    r = rotation.rand;
    report('SO3 rotate L=128', timeit(@() rotate(F,r),3));
    F.bandwidth = 64;
    report('SO3 rotate L=64', timeit(@() rotate(F,r),3));
  case 's2'
    L = 128;
    sF = S2FunHarmonic(randn((L+1)^2,1)+1i*randn((L+1)^2,1));
    sF.isReal = true;
    v = vector3d.rand(1e6);
    report('S2 eval L=128 J=1e6', timeit(@() sF.eval(v),2));
    vals = rand(1e6,1);
    report('S2 quadrature L=128 J=1e6', timeit(@() S2FunHarmonic.quadrature(v,vals,'bandwidth',L),2));
    for L = [256 512 1024]
      g = quadratureS2Grid(L,'ClenshawCurtis');
      sF = S2FunHarmonic(randn((L+1)^2,1)+1i*randn((L+1)^2,1));
      report(sprintf('S2 eval Clenshaw-Curtis L=%d',L), timeit(@() sF.eval(g),2));
    end
    f = @(v) exp(v.z);
    report('S2 quadrature of a callable L=256', timeit(@() S2FunHarmonic.quadrature(f,'bandwidth',256),2));
    sF = S2FunHarmonic(randn(129^2,1));
    report('S2 symmetrise m-3m L=128', timeit(@() symmetrise(sF,cub),3));
    v = vector3d.rand(1e5);
    report('S2 calcDensity 1e5', timeit(@() calcDensity(v),2));
end
end

function F = randomODF(L,cs,ss)
  d = deg2dim(L+1);
  F = SO3FunHarmonic(randn(d,1)+1i*randn(d,1),cs,ss);
  F.isReal = true;
end

function t = timeit(f,reps)
  f();
  t = inf;
  for i = 1:reps
    tic; f(); t = min(t,toc);
  end
end

function report(name,t)
  fprintf('%-48s %9.1f ms\n',name,1000*t);
end
