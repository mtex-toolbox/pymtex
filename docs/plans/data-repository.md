# The joint data repository

One repository of sample data for the three ports, `mtex-toolbox/mtex-data`, public. The
files and one manifest are shared; the recipes (phases, Euler corrections, crops a page
applies) stay in each port. Decided with the user on 2026-09-26.

## Layout

```
mtex-data/
  README.md            what is here, how a port pins a version, the licence
  LICENSE              the licence of the data
  manifest.json        generated: every file and every data set
  tools/manifest.py    writes manifest.json from the tree and large.json, checks it
  large.json           the files kept as release assets: path, sha256, size, release tag
  EBSD/ EBSD3/ PoleFigure/ ODF/ orientation/ tensor/ vector3d/ Neper/ VPSC/ cif/ grain2d/
```

The folders and paths are MATLAB's `data/`, so a path in a port's manifest stays the same
and a local copy of MATLAB's `data/` still serves. MATLAB's cached `.mat` files in the root
of `data/` (`forsterite.mat`, ...) are `mtexdata`'s own caches and are left out.

## Large files

A file of 50 MiB or more is not committed. It is attached to a GitHub release and listed in
`large.json`: `SmallIN100_MeshStats.dream3d` (176 MB), `copper29.h5oina` (157 MB),
`trueEbsdWCCo.mat` (172 MB, now on Zenodo 13870131), `EMSphinx.h5` (48 MiB, just below, stays
in git). An asset keeps the URL of the release it was first attached to; a later release
does not upload it again.

## The manifest

```json
{
  "version": "v1",
  "files": {
    "EBSD/Forsterite.ctf": {"sha256": "...", "size": 13925969,
                            "url": "https://raw.githubusercontent.com/mtex-toolbox/mtex-data/v1/EBSD/Forsterite.ctf"}
  },
  "datasets": {
    "forsterite": {"type": "EBSD", "description": "...", "files": ["EBSD/Forsterite.ctf"]}
  }
}
```

A committed file's URL is the raw URL at the tag, a large one the release asset. The data
sets are MATLAB's `summary.txt` joined with the Julia and Python data set lists (the names
lower case, as both ports look them up); a `.cpr` brings its `.crc`.

## The ports

- Python: `mtex/io/sampledata.json` is `manifest.json` at a tag, copied; `sampledata.py`
  reads its `files` and `datasets`.
- Julia: `tools/generate_sample_artifacts.jl` reads `manifest.json` at a tag instead of the
  MTEX git revision; `Artifacts.toml` keeps its tree hashes.
- MATLAB: `tools/mtexdata.m` changes in a later release; `mtex/data` stays until then.

## Steps

1. Done (2026-09-26): the tree in `../mtex-data`, MATLAB's `data/` at 93203281 without
   `summary.txt` and the misspelt `Neeper/`, the files the ports read added
   (`EBSD/trueEbsdWCCoSmall.h5`, `EBSD/trueEbsdWCCo.mat`, `EBSD/copper29.h5oina`, the Neper
   tessellations and `lamnormals.txt`, `ori_in.txt`), 223 files, 45 data sets in
   `datasets.json`, `tools/manifest.py`, CC BY 4.0; committed and tagged `v1` locally.
2. Done: `mtex-toolbox/mtex-data` created, `main` and `v1` pushed, the release `v1` with the
   three files of `large.json` attached; every URL of the manifest checked.
3. Done: `mtex/io/sampledata.json` is the repository's `manifest.json` at `v1`; `mtexdatafile`
   gives the files of every data set of the repository, those without a recipe here too
   (`my100grains`, which the Neper pages read).
4. Done: `tools/generate_sample_artifacts.jl DATA_REPO TAG` reads the manifest at the tag;
   the tree hashes of `Artifacts.toml` are unchanged, only the URLs moved.
5. MATLAB's `mtexdata` in a later release, reading `manifest.json` with `jsondecode`.

Updating a port: `cp ../mtex-data/manifest.json mtex/io/sampledata.json` at the tag, and
`julia --project=. tools/generate_sample_artifacts.jl ../mtex-data <tag>` in the Julia port.

## Decided

- The licence: CC BY 4.0, the README asking to cite the sources the pages give.
- Everything MATLAB's `data/` tracks moves, the MATLAB-only files included.
