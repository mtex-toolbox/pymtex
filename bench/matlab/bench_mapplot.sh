#!/usr/bin/env bash
# the MATLAB side of bench/mapplot.py through the bridge of ../frameSymmetry (SESSION_NAME=mtexfs), one map per call
cd /home/hielscher/mtex
B=frameSymmetry/docs/agents/matlab-bridge
O=$PWD/pymtex/bench/out
for f in ${MAPS:-frameSymmetry/data/EBSD/Forsterite.ctf frameSymmetry/data/EBSD/Emsland_plessite_500x_15.ctf $O/forsterite_2x2.ctf $O/forsterite_4x4.ctf}; do
  n=$(basename "$f" .ctf)
  case "$f" in /*) ;; *) f="$PWD/$f" ;; esac
  SESSION_NAME=mtexfs timeout 300 $B/.venv/bin/python $B/matlab_run.py "addpath('$PWD/pymtex/bench/matlab'); bench_mapplot('$f', '$O/mapplot_matlab_$n.json')" > "$O/mapplot_matlab_$n.log" 2>&1
  cat "$O/mapplot_matlab_$n.json" 2>/dev/null; echo " $n"
done
