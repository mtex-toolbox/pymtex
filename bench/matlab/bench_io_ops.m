% the operations of bench/io_ops.py in MATLAB
function bench_io_ops(out, ops, K)

% the files of one data directory for every MTEX version, whose own may not hold them
data = fullfile(getenv('HOME'), 'mtex', 'frameSymmetry', 'data');
files = struct('load_ctf', fullfile(data,'EBSD','Forsterite.ctf'), 'load_ang', fullfile(data,'EBSD','ferrite.ang'), ...
  'load_osc', fullfile(data,'EBSD','olivineopticalmap.ang'), 'load_1C_1', '/home/hielscher/mtex/data/1C_1.ctf', ...
  'load_ang_SAM', '/home/hielscher/mtex/data/SAM-1-X-clean.ang', 'load_ang_PAGR', '/home/hielscher/mtex/data/PAGR.ang');
copper29 = fullfile(getenv('HOME'), '.mtex', 'data', 'EBSD', 'copper29.h5oina');
edax = '/home/hielscher/mtex/data/h5_edax_2datasets.edaxh5';
d = dir(fullfile(data,'PoleFigure','dubna','*.cnv'));
pfFiles = fullfile({d.folder}, {d.name});
cs = crystalSymmetry('321', [4.916 4.916 5.4054], 'mineral', 'Quartz');

% the map the export writes, read and turned only when the export is asked for
turned = [];
if any(strcmp('export_h5', ops))
  try, turned = rotate(EBSD.load(copper29), rotation.byAxisAngle(vector3d(1,1,1), 10*degree), 'keepXY'); catch, end
end
written = fullfile(out, 'matlab_export_h5.h5oina');

T = cell(0, 3);
for f = fieldnames(files).'
  T(end+1,:) = {f{1}, @() EBSD.load(files.(f{1})), @rows}; %#ok<AGROW>
end
T(end+1,:) = {'load_h5oina_copper29', @() EBSD.load(copper29), @rows};
T(end+1,:) = {'load_h5_edax', @() EBSD.load(edax, 'dataSet', 1), @rows};
T(end+1,:) = {'export_h5', @() exported(turned, written), @(f) rows(EBSD.load(f))};
T(end+1,:) = {'load_dubna', @() PoleFigure.load(pfFiles, cs, 'superposition', {1,1,[0.52 1.23],1,1,1,1}), @pfRows};
algo = containers.Map(T(:,1), [repmat({'EBSD import'}, 1, 8), {'EBSD export'}, {'pole figure import'}]);
crosscheckRun(out, T, ops, algo, K, {}, containers.Map({'load_1C_1', 'load_ang_PAGR'}, {1, 1}));

end


% the export into this port's file, which the rows read back
function fname = exported(ebsd, fname)
export(ebsd, fname, 'silent');
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
