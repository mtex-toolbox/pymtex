% the operations of bench/plotting_ops.py in MATLAB: one result per symmetry, written for both
% of Python's variants
function bench_plotting_ops(out, ops, K)

O = reshape(fread(fopen(fullfile(out,'O.bin')),'double'),4,[]).'; fclose('all');
q = quaternion(O(:,1),O(:,2),O(:,3),O(:,4));
T = cell(0, 3);
for s = {'m3m', '6mmm'; 'm-3m', '6/mmm'}
  ori = orientation(q, crystalSymmetry(s{2}));
  f = @() ipfHSVKey(ori.CS).orientation2color(ori);
  T(end+1,:) = {['ipf_' s{1}], f, @(c) c}; %#ok<AGROW>
  T(end+1,:) = {['ipf_exact_' s{1}], f, @(c) c}; %#ok<AGROW>
end
crosscheckRun(out, T, ops, containers.Map(T(:,1), repmat({'map plotting'}, size(T,1), 1)), K);

end
