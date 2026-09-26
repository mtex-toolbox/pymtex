function probe_gbnd3d()
% the three forms of mtex/analysis/gbnd3d.py on the IN100 mesh
g = grain3d.load(fullfile(mtexDataPath,'EBSD','SmallIN100_MeshStats.dream3d'));
gB = g.boundary(all(g.boundary.phaseId == 2,2));
v = vector3d([0 1 1 0.3],[0 0 1 0.5],[1 0 1 0.8]);
f = calcGBND(gB); fprintf('specimen %s\n', mat2str(f.eval(v).',7));
% master's loader reads DREAM.3D's (x,y,z,w) as (a,b,c,d): the rotations corrected to (w,-x,-y,-z)
q = g.prop.meanRotation; g.prop.meanRotation = rotation(quaternion(q.d,-q.a,-q.b,-q.c));
cs = g.CS;
c = calcGBND(gB,g); fprintf('crystal %s\n', mat2str(c.eval(Miller(v,cs)).',7));
m = orientation.byAxisAngle(vector3d(1,1,1),60*degree,cs,cs);
d = calcGBND(gB,g,m); fprintf('gbcd %s\n', mat2str(d.eval(Miller(v,cs)).',7));
end
