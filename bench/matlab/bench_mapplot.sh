#!/usr/bin/env bash
# the MATLAB side of bench/mapplot.py through the bridge of ../frameSymmetry (SESSION_NAME=mtexfs), one map per call
cd /home/hielscher/mtex
B=frameSymmetry/docs/agents/matlab-bridge
O=$PWD/mtexPython/bench/out
for f in frameSymmetry/data/EBSD/Forsterite.ctf frameSymmetry/data/EBSD/Emsland_plessite_500x_15.ctf $O/forsterite_2x2.ctf; do
  n=$(basename "$f" .ctf)
  SESSION_NAME=mtexfs timeout 600 $B/.venv/bin/python $B/matlab_run.py "addpath('$PWD/mtexPython/bench/matlab'); bench_mapplot('$PWD/$f', '$O/mapplot_matlab_$n.json')" > "$O/mapplot_matlab_$n.log" 2>&1
  cat "$O/mapplot_matlab_$n.json" 2>/dev/null; echo " $n"
done
