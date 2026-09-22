#!/usr/bin/env bash
# every EBSD file of ../mtex-paper read by MTEX through the bridge of ../frameSymmetry and by the port, compared;
# start the bridge first: (cd ../frameSymmetry && SESSION_NAME=mtexfs docs/agents/matlab-bridge/start_session.sh)
# and drop master from its path: rmpath(genpath(...master)); clear classes; rehash
cd /home/hielscher/mtex
S=${PROBE_OUT:-/tmp/mtex-probe}; mkdir -p "$S"
B=frameSymmetry/docs/agents/matlab-bridge
find mtex-paper -type f \( -iname "*.ctf" -o -iname "*.cpr" -o -iname "*.h5" \) -not -path "*/.git/*" -not -name "K-SX.ctf" -not -name "Quartz-175B_05.ctf" | sort | while read -r f; do
  n=$(basename "$f")
  SESSION_NAME=mtexfs $B/.venv/bin/python $B/matlab_run.py "addpath('$PWD/pymtex/bench/matlab'); probe_ebsd('$PWD/$f', '$S/m_$n.json')" > "$S/mlog_$n.txt" 2>&1
  pymtex/.venv/bin/python pymtex/bench/matlab/probe_python.py "$f" "$S/p_$n.json" > "$S/plog_$n.txt" 2>&1
  pymtex/.venv/bin/python pymtex/bench/matlab/compare.py "$S/m_$n.json" "$S/p_$n.json" "$n" 2>&1 || { echo "$n: probe failed"; tail -2 "$S/mlog_$n.txt" "$S/plog_$n.txt"; }
done
echo ALLDONE
