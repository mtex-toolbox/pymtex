% the operations of bench/embedding_ops.py in MATLAB
function bench_embedding_ops(out, ops, K)

ld = @(name) reshape(fread(fopen(fullfile(out,[name '.bin'])),'double'),4,[]).';
O = ld('O'); P = ld('P'); C = ld('C');
fclose('all');
q = @(X) quaternion(X(:,1),X(:,2),X(:,3),X(:,4));
NP = 1e4;

quat = @(o) [o.a(:) o.b(:) o.c(:) o.d(:)];
col = @(x) x(:);
T = cell(0, 3);
for s = {'m3m', '6mmm'; 'm-3m', '6/mmm'}
  cs = crystalSymmetry(s{2});
  oO = orientation(q(O), cs); oP = orientation(q(P), cs); oC = orientation(q(C), cs);
  T(end+1,:) = {['embed_' s{1}], @() embed(oO, cs), col}; %#ok<AGROW>
  T(end+1,:) = {['project_' s{1}], @() orientation((embedding(oO(1:NP)) + embedding(oP(1:NP))) ./ 2), quat}; %#ok<AGROW>
  T(end+1,:) = {['mean_' s{1}], @() orientation(mean(embedding(oC))), quat}; %#ok<AGROW>
end
algo = containers.Map(T(:,1), repmat({'orientation embedding'}, size(T,1), 1));
reps = containers.Map({'project_m3m', 'project_6mmm'}, {1, 1});
crosscheckRun(out, T, ops, algo, K, {}, reps);

end


% the distance of every image to the first, over the norm of the identity's image
function d = embed(o, cs)
e = embedding(o);
d = norm(e - embedding(o(1))) ./ norm(embedding.id(cs));
end
