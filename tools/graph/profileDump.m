function profileDump(fname, algorithm, port, caseName, wall, cores)
% the profile of the last profile on/off in the record format of tools/graph/profiling.py, with
% the cores the timed runs kept busy and the computational threads MATLAB was given

p = profile('info');
ft = p.FunctionTable;
recs = cell(numel(ft),1);
for k = 1:numel(ft)
  ch = ft(k).Children;
  children = zeros(numel(ch),2);
  for j = 1:numel(ch)
    children(j,:) = [ch(j).Index - 1, ch(j).TotalTime];
  end
  selfTime = ft(k).TotalTime - sum(children(:,2));
  recs{k} = struct('file',ft(k).FileName,'line',0,'name',ft(k).FunctionName, ...
    'self',max(selfTime,0),'total',ft(k).TotalTime,'calls',ft(k).NumCalls,'children',{num2cell(children,2)});
end
if nargin < 6, cores = []; end
s = struct('algorithm',algorithm,'port',port,'case',caseName,'wall',wall,'cores',cores, ...
  'threads',maxNumCompThreads,'functions',{recs});
fid = fopen(fname,'w');
fwrite(fid,jsonencode(s));
fclose(fid);

end
