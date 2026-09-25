% where MATLAB's calcODF spends its time on the Dubna pole figures
pf = mtexdata('dubna');
odf = calcODF(pf,'silent');
tic; odf = calcODF(pf,'silent'); fprintf('calcODF %.2f s\n',toc);
solver = MLSSolver(pf); solver.init;
fprintf('grid %d orientations, kernel bandwidth %d, halfwidth %.2f deg\n', length(solver.S3G), solver.psi.bandwidth, solver.psi.halfwidth/degree);
for i = 1:pf.numPF
  nh = 0;
  for j = 1:length(pf.allH{i}), nh = nh + length(unique(symmetrise(pf.allH{i}(j)),'noSymmetry','antipodal')); end
  fprintf('pf %d: gh nodes %d, r nodes %d\n', i, length(solver.S3G)*nh, length(solver.SS.properGroup*pf.allR{i}));
end
profile clear; profile on; odf = calcODF(pf,'silent'); profile off;
p = profile('info'); ft = p.FunctionTable;
[~,ix] = sort([ft.TotalTime],'descend');
for k = ix(1:22), fprintf('%-60s calls %5d total %7.3f self %7.3f\n', ft(k).FunctionName, ft(k).NumCalls, ft(k).TotalTime, ft(k).TotalTime - sum([ft(k).Children.TotalTime])); end
