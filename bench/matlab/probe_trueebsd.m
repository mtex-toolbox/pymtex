function probe_trueebsd(what, inFile, outFile)
% the MATLAB side of the M11 pins: xcfShift on two images of a file, or the trueEbsd page run
% on trueEbsdWCCoSmall, written to a .mat file for tests/test_registration.py and the pages
switch what
  case 'xcf'
    % A, B and the options of one xcfShift call, from a .mat file
    d = load(inFile);
    [u,peak,pos] = xcfShift(d.A, d.B, 'ROISize', d.roi, 'numROI', d.numROI(:).');
    ux = u.x; uy = u.y; px = pos.x; py = pos.y; %#ok<NASGU>
    save(outFile, 'ux', 'uy', 'peak', 'px', 'py', '-v7');
  case 'page'
    % the EBSDTrueEbsd page up to undistort, with the shifts and the fitted hops
    plottingConvention.default('y↓→x');
    ebsd = mtexdata('trueEbsdWCCoSmall','silent');
    img = ebsd.opt.trueEbsdImgs;
    imgList = [mapImage(ebsd.bc, ebsd, 'name','bcImg'), ...
      mapImage(img.fsdB3, 'dxy',img.pixSzImg, 'name','fsdB3'), ...
      mapImage(img.fsdT3, 'dxy',img.pixSzImg, 'name','fsdT3'), ...
      mapImage(img.fsdT1, 'dxy',img.pixSzImg, 'name','fsdT1'), ...
      mapImage(img.fsdT10,'dxy',img.pixSzImg, 'name','fsdT10')];
    imgList(2:end) = rescale(imboxfilt(imgList(2:end),3));
    T = [spatialTransformShift + spatialTransformDrift, spatialTransformId, spatialTransformShift, spatialTransformTilt];
    job = trueEbsd(imgList,T);
    tic; job.pixelSizeMatch; tPSM = toc; %#ok<NASGU>
    job.setOptions('numROI',16);
    tic; job.calcDistortion('fitErr'); tCD = toc; %#ok<NASGU>
    tic; job.undistort; tU = toc; %#ok<NASGU>
    out = struct();
    for n = 1:numel(job.resizedList)
      out.(sprintf('resized%d',n)) = job.resizedList(n).img;
      out.(sprintf('undist%d',n)) = job.undistortedList(n).img;
    end
    for n = 1:numel(job.shifts)
      for m = 1:numel(job.shifts{n})
        ps = job.shifts{n}(m);
        if isempty(ps.u), continue; end
        out.(sprintf('s%d_%d',n,m)) = [ps.pos.x ps.pos.y ps.u.x ps.u.y ps.peak];
      end
      ps = job.fitError(n);
      if ~isempty(ps.u), out.(sprintf('e%d',n)) = [ps.pos.x ps.pos.y ps.u.x ps.u.y ps.peak]; end
      out.(sprintf('opt%d',n)) = [job.opt(n).roiSize(:).' job.opt(n).edgeWidth];
    end
    o = job.undistortedList(1).ebsd;
    out.phaseId = o.phaseId; out.q = [o.rotations.a o.rotations.b o.rotations.c o.rotations.d];
    out.x = o.pos.x; out.y = o.pos.y;
    out.times = [tPSM tCD tU];
    save(outFile, '-struct', 'out', '-v7');
end
end
