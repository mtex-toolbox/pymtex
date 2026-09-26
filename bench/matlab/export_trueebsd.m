function export_trueebsd(outFile)
% mtexdata('trueEbsdWCCoSmall') as an HDF5 file of plain arrays, for the Python mtexdata recipe:
% the gridded EBSD map (positions, phases, quaternions, properties), its phase list, and the four
% forescatter images with their pixel size, as ebsd.opt.trueEbsdImgs holds them
if nargin < 1, outFile = fullfile(getenv('HOME'),'.mtex','data','EBSD','trueEbsdWCCoSmall.h5'); end
if isfile(outFile), delete(outFile); end
ebsd = mtexdata('trueEbsdWCCoSmall','silent');
w = @(name,v) writeOne(outFile,name,v);
w('/ebsd/x', ebsd.pos.x); w('/ebsd/y', ebsd.pos.y);
w('/ebsd/phaseId', double(ebsd.phaseId));
q = ebsd.rotations;
w('/ebsd/rotation', cat(3, q.a, q.b, q.c, q.d));
for fn = fieldnames(ebsd.prop).'
  w(['/ebsd/prop/' fn{1}], double(ebsd.prop.(fn{1})));
end
h5writeatt(outFile,'/ebsd','scanUnit',ebsd.scanUnit);
h5writeatt(outFile,'/ebsd','how2plot',char(ebsd.how2plot));
for k = 1:numel(ebsd.CSList)
  cs = ebsd.CSList(k); g = sprintf('/phases/%d', k-1);
  w([g '/index'], k-1);
  h5writeatt(outFile, g, 'mineral', char(cs.mineral));
  if isa(cs,'crystalFrame')
    h5writeatt(outFile, g, 'pointGroup', char(cs.pointGroup));
    h5writeatt(outFile, g, 'axes', [norm(cs.aAxis) norm(cs.bAxis) norm(cs.cAxis)]);
    h5writeatt(outFile, g, 'angles', [cs.alpha cs.beta cs.gamma] / degree);
    h5writeatt(outFile, g, 'alignment', char(strjoin(cs.alignment,', ')));
  end
end
im = ebsd.opt.trueEbsdImgs;
for fn = {'fsdB3','fsdT3','fsdT1','fsdT10'}
  w(['/images/' fn{1}], im.(fn{1}));
end
w('/images/pixSzImg', im.pixSzImg);
fprintf('wrote %s\n', outFile);
end

function writeOne(f,name,v)
h5create(f,name,size(v),'Datatype','double');
h5write(f,name,double(v));
end
