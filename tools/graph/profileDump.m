function profileDump(fname, algorithm, port, caseName, wall)
% the profile of the last profile on/off in the record format of tools/graph/profiling.py

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
s = struct('algorithm',algorithm,'port',port,'case',caseName,'wall',wall,'functions',{recs});
fid = fopen(fname,'w');
fwrite(fid,jsonencode(s));
fclose(fid);

end
