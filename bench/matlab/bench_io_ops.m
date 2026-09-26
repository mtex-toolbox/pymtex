% the operations of bench/io_ops.py in MATLAB
function bench_io_ops(out, ops, K)

data = fullfile(mtex_path, 'data');
files = struct('load_ctf', fullfile(data,'EBSD','Forsterite.ctf'), 'load_ang', fullfile(data,'EBSD','ferrite.ang'), ...
  'load_osc', fullfile(data,'EBSD','olivineopticalmap.ang'), 'load_1C_1', '/home/hielscher/mtex/data/1C_1.ctf');
d = dir(fullfile(data,'PoleFigure','dubna','*.cnv'));
pfFiles = fullfile({d.folder}, {d.name});
cs = crystalSymmetry('321', [4.916 4.916 5.4054], 'mineral', 'Quartz');

T = cell(0, 3);
for f = fieldnames(files).'
  T(end+1,:) = {f{1}, @() EBSD.load(files.(f{1})), @rows}; %#ok<AGROW>
end
T(end+1,:) = {'load_dubna', @() PoleFigure.load(pfFiles, cs, 'superposition', {1,1,[0.52 1.23],1,1,1,1}), @pfRows};
algo = containers.Map(T(:,1), [repmat({'EBSD import'}, 1, 4), {'pole figure import'}]);
crosscheckRun(out, T, ops, algo, K, {}, containers.Map({'load_1C_1'}, {1}));

end


% (x, y, phase, a, b, c, d), the phase the index into the phase list counted from 0 for notIndexed
function R = rows(e)
ph = double(e.phaseId(:)) - strcmp(e.mineralList{1}, 'notIndexed');
ph(~e.isIndexed) = 0;
R = [e.pos.x(:) e.pos.y(:) ph e.rotations.a(:) e.rotations.b(:) e.rotations.c(:) e.rotations.d(:)];
end


function R = pfRows(pf)
R = [];
for k = 1:pf.numPF
  r = pf.allR{k}; I = pf.allI{k};
  R = [R; r.x(:) r.y(:) r.z(:) I(:)]; %#ok<AGROW>
end
end
