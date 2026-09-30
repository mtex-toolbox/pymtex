# Subplan P1 - geometry

Status: G0 to G7 done. Syntax reference:
`../frameSymmetry/geometry/` (the MATLAB branch that already spells ADR 0008). Design
reference: `../mtexJulia/lib/MTEXGeometry/`, its `docs/plans/geometry.md`,
`docs/design/semantics.md` and ADR 0001 to 0011. Tests: `../frameSymmetry/tests/core/`
and `../mtexJulia/test/`.

## 1. Deliverable

A package `mtex.geometry` whose public layer reads like the MATLAB of the frameSymmetry
branch and whose inside is the Julia design on NumPy arrays, in float64 and float32:

- `vector3d` with polar coordinates, antipodal axes, normalisation; `Miller` indices in
  both bases through the crystal frame;
- `quaternion`, `rotation`, the improper elements, every representation MTEX offers
  (`byEuler` in the five conventions, `byAxisAngle`, `byMatrix`, `byRodrigues`,
  `byHomochoric`, `map`, `fit`);
- `symmetry` (the point group from its Hermann-Mauguin symbol), `crystalFrame`,
  `specimenFrame`, `notIndexedFrame` with the session register, `plottingConvention`;
- `orientation` with `CS`/`SS`, misorientations with grain exchange, symmetry-reduced
  `dot`, `angle`, `unique`, `mean`, `symmetrise`, `project2FundamentalRegion`,
  `fundamentalRegion`, `transformReferenceFrame`, the standard orientations and
  orientation relationships, `variants`;
- `sphericalRegion` and `fundamentalSector`, `orientationRegion`;
- `S1Grid`, `S2Grid`, `SO3Grid`, `equispacedS2Grid`, `regularS2Grid`, `fibonacciS2Grid`,
  `equispacedSO3Grid`, `localOrientationGrid`, `localS2Grid`;
- `fibre`, `plane3d`, `slipSystem`, `dislocationSystem`, `crystalShape`,
  `SO3TangentVector`, `calcParent2Child`, `calcVariantId`, `calcParent`;
- the geometry tests of `tests/core` ported and passing.

Plotting is P5; the plot methods of the geometry classes are added there.

## 2. The two references, and the rule between them

The frameSymmetry branch says **what the user writes**: class names, constructor forms,
property names, verb names, flag names. MTEX.jl says **what is inside**: which concepts
exist, what each carries, the kernels, the numerics, what MATLAB does that is a
workaround and is not ported. When the two disagree on a name the user sees, MATLAB wins;
when they disagree on a representation or an algorithm, Julia wins.

What the user sees, taken from MATLAB:

- Class names in MATLAB casing: `vector3d`, `quaternion`, `rotation`, `orientation`,
  `referenceFrame`, `crystalFrame`, `specimenFrame`, `notIndexedFrame`, `symmetry`,
  `plottingConvention`, `S2Grid`, `SO3Grid`, `sphericalRegion`, `orientationRegion`,
  `fibre`, `plane3d`, `slipSystem`, `crystalShape`. `Miller(...)` is a function that
  returns a `vector3d` in a crystal frame, as on the branch. `crystalSymmetry` and
  `specimenSymmetry` do not exist.
- Static constructors as class methods: `rotation.byEuler(...)`, `orientation.byMiller`,
  `vector3d.byPolar`, `vector3d.X`, `vector3d.rand(n)`, `specimenFrame.rolling`,
  `orientation.cube(cs)`.
- Properties as properties: `v.x`, `v.theta`, `v.antipodal`, `m.hkl`, `m.uvw`,
  `m.dispStyle`, `rot.phi1`, `ori.CS`, `ori.SS`, `cs.Laue`, `cs.properGroup`, `cs.rot`,
  `cs.abc`, `cs.mineral`, `cs.color`, `sF.how2plot`. What MATLAB declares as a property,
  and every `is*` predicate, is a Python property; what MATLAB implements as a method
  (`angle`, `axis`, `matrix`, `norm`) is a Python method with parentheses, `rot.angle()`,
  since Python cannot call a method without them.
- Verbs as functions and as methods: `angle(o1, o2)` and `o1.angle(o2)` are the same
  call; `dot`, `cross`, `norm`, `mean`, `unique`, `symmetrise`, `inv`, `log`, `exp`,
  `Euler`, `matrix`, `axis`, `rotate`, `project2FundamentalRegion` all exist in both forms
  and are exported by `mtex`. The method is the implementation, the function a
  one-line wrapper.
- The `_outer` names stay as conveniences (`dot_outer`, `angle_outer`), one line each
  over broadcasting, because every MTEX script uses them.
- `degree` is a constant; angles are radians.

What the user sees, taken from Python because MATLAB has no counterpart:

| MATLAB | Python |
| --- | --- |
| `v(3)`, `v(1:10)`, `v(end)`, `v(mask)` | `v[2]`, `v[0:10]`, `v[-1]`, `v[mask]`, zero-based |
| `v(:)`, `size(v)`, `numel(v)`, `length(v)`, `isempty(v)` | `v.flatten()`, `v.shape`, `v.size`, `len(v)`, `v.size == 0` |
| `[v1, v2]`, `[v1; v2]`, `cat(dim, ...)` | `cat(v1, v2)`, `cat(v1, v2, axis=0)` |
| `v'`, `reshape(v, ...)`, `repmat` | `v.T`, `v.reshape(...)`, NumPy broadcasting |
| `a * b`, `a .* b` | `a * b`; broadcasting covers elementwise, outer and scalar cases |
| `dot(m1, m2, 'noSymmetry')`, `symmetrise(m, 'unique')` | `dot(m1, m2, symmetry=False)`, `symmetrise(m, unique=True)` |
| `vector3d(x, y, z, 'antipodal')` | `vector3d(x, y, z, antipodal=True)` |
| `crystalFrame('m-3m', [4 4 4], 'mineral', 'Fo')` | `crystalFrame('m-3m', [4, 4, 4], mineral='Fo')` |
| `getMTEXpref('EulerAngleConvention')` | `getMTEXpref('EulerAngleConvention')`, unchanged |

The rule for strings: a positional string **names** something, a boolean option is a
**keyword**. `Miller(u, v, w, cs, 'uvw')`, `rotation.byEuler(a, b, c, 'ZYZ')`,
`Miller('{111}', cs)`, `crystalFrame([3, 3, 5], 'X||a')` stay as they are;
`'antipodal'`, `'noSymmetry'`, `'unique'`, `'proper'` become `antipodal=True` and so on.
Option names keep MTEX's camelCase.

## 3. The Julia decisions and their Python spelling

The Julia port encoded most distinctions as types because Julia's inference needs it.
Python has no inference to protect, so the rule of `docs/PLAN.md` D2 applies: a
distinction that changes *which algorithm runs* is a class, one that changes *a parameter
of the same algorithm* is a field. Under the MATLAB syntax the table is:

| Julia | Python | kept from Julia |
| --- | --- | --- |
| `Axis` type (ADR 0002) | `v.antipodal` field, as MATLAB | the semantics: `dot`, `angle`, `==`, `unique`, `mean` are antipodal when either operand is; arithmetic clears the flag; `Miller(..., antipodal=True)` builds the direction in the Laue sibling of its frame and stores no flag |
| `Rotation` proper, `OrthogonalTransform` improper (ADR 0003) | two classes, `rotation` and `orthogonalTransform` | `rotation` has no `i` field; `-rot`, `rotation.inversion`, `reflection(n)` return the O(3) class; `cs.rot` is the full element list in the O(3) class, `cs.properGroup.rot` the rotations; `symmetrise(ori)` returns proper equivalents only |
| frames as interned handles (ADR 0004) | the same: `==` is `is`, a register keyed on the ADR 0008 key, sealed after registration except `name`, `color`, `how2plot` | everything, including the siblings `Laue`, `properGroup`, `properSubGroup`, `fullSym`, `stripSym` minted through the register |
| frame kinds as type parameters (ADR 0005) | the class of the frame handle; `ori.frameA`, `ori.frameB` fields with `CS`/`SS` positional | products check the kinds at run time with a message |
| index basis as type parameter (ADR 0006) | `v.dispStyle` field, `'hkl'` or `'uvw'`, as the branch has it | storage stays Cartesian; `hkil`/`UVTW` derived at read time; `cross` flips the basis; `round` rounds in the basis; an array does not mix bases |
| `UnitVec3` type (ADR 0007) | `v.isNormalized` field, as MATLAB | `theta`, `rho`, `angle`, the projections skip normalisation when set; `normalize` sets it; `+`, `mean`, `cross` clear it |
| grain exchange as type parameter (ADR 0008) | `ori.antipodal` field, as MATLAB | may be `True` only when `frameA is frameB`, checked at construction; every reduction has the exchange branch |
| join rule (ADR 0009) | the same, in `cat` and in every binary operation | identical handle, frame-free side, sibling by group intersection, else an error naming `transformReferenceFrame` |
| tangent side as type (ADR 0010) | `SO3TangentVector` with a `side` field, `'left'` or `'right'` | the frame is derived from the reference, `log` reduces the pair first |
| one type per concept (ADR 0011) | one class per concept; a single element is the zero-dimensional container | no scalar classes |
| point groups from the symbol, closure order, nothing indexed by position | the same | `symmetry('m-3m')` builds the closure; the name table only names; section 8 of the Julia subplan is the checklist |
| kernels: non-normalising product, chord angle, `max_k |<q1, l q2 r>|` | the same, in NumPy | the tolerance rule: compare the dot or the chord, never `acos` in a loop |

Dropped from MATLAB, as the Julia port dropped them: `dynOption` and `v.opt`, `subSet`,
`repmat` and `reshape` overloads beyond NumPy's, `numArgumentsFromSubscript`,
`check_option` and the option-string parser, `isCrystalDirection` and `hasSymmetry` as
predicates a user calls (they become methods on the frame), `rotate_outer`, `cross_outer`,
`quat_dot_outer`, the `sealed` flag and `clone`, the two persistent registers, the
memoised region, `loadobj`, `dot_angle`, `qq`, `qmatrix` as public names.

Mutability: geometry objects allow the property writes MATLAB scripts make
(`v.antipodal = True`, `ori.CS = cs`, `m.dispStyle = 'uvw'`, `cs.color = 'red'`) and they
mutate in place, which is how NumPy and orix behave too. Every operation returns a new
object, slicing returns a view where NumPy would. `copy()` exists. The pages say once that
`ori2 = ori` shares, unlike MATLAB. **Discussion.**

## 4. The array model

One base class, `mtex.geometry._array.MtexArray` (internal name), gives every
container `shape`, `ndim`, `size`, `__len__`, `__getitem__` returning the same class,
`__setitem__`, `__iter__`, `reshape`, `flatten`, `T`, `copy`, `cat`, and the broadcasting
of binary operations over the trailing coordinate axis. A subclass declares its
coordinate arrays and its carried attributes (frames, flags); the base does the rest.

- `vector3d` holds one `(..., 3)` float array; `x`, `y`, `z` are views of its columns.
  `antipodal`, `isNormalized`, `dispStyle` and the frame handle are carried attributes.
- `quaternion` holds one `(..., 4)` array; `a`, `b`, `c`, `d` are views. `rotation` is
  a `quaternion` subclass. `orientation` is a `rotation` subclass carrying `frameA`,
  `frameB`, `antipodal`.
- `orthogonalTransform` holds a `(..., 4)` array and a `(...)` parity array.
- Frames and point groups are ordinary objects; a point group holds its elements as one
  `orthogonalTransform` array and caches the proper subset.
- `dtype` is float64 unless the input is float32; frames and groups are float64 and
  converted on use.

Binary operations `*`, `+`, `-`, `dot`, `angle`, `cross` broadcast the leading shapes as
NumPy does, so `rot[:, None] * v[None, :]` is the outer product and the `_outer` names
are that one line. Comparison with symmetry reduces over the group with one `einsum` per
group element block, vectorised over the whole array; E1 in `docs/PLAN.md` measures it.

## 5. Order of work

**G0 - the semantics document and the array base. DONE 2026-09-21** (`docs/design/semantics.md`, `tests/test_semantics.py` with S1 to S26 skipped until their classes exist, `tests/test_oracles.py` with the S9 and S10 formulas passing against SciPy now, `mtex/geometry/_array.py` with `tests/test_array.py`; `pyproject.toml`, a uv environment on Python 3.14 with NumPy 2.5, SciPy 1.18, orix 0.15 as a test dependency). `docs/design/semantics.md` is the
Julia document S1 to S26 respelled in the Python syntax, statement for statement;
`tests/test_semantics.py` holds one test per statement, the orientation and frame
statements skipped until G4/G5; `tests/test_oracles.py` checks products, Euler and
matrix conversions against `scipy.spatial.transform.Rotation` and orix, test
dependencies only. `_array.py` is written and tested with a dummy subclass before any
geometry class exists: indexing, views, `cat`, broadcasting, `float32`.

**G1 - the frame-free layer. DONE 2026-09-21** (`vector3d.py`, `quaternion.py` with `quaternion`, `rotation`, `orthogonalTransform` and `reflection`, `representations.py` and `homochoric.py` as kernels on coefficient arrays, `plottingconvention.py` with the session default and `plotx2east` and friends, `units.py`, `display.py`, `session.py` with the preferences, `funcs.py` with the verbs as functions; S1 to S18 pass, 86 tests, 8 skipped for G4/G5; E1 numbers in `docs/research/benchmark-g1.md`; orix's Bunge rotation is the inverse of MTEX's, recorded in `tests/test_oracles.py`). The parity of an orthogonal transform is a fifth coordinate, so the container base serves it unchanged; the class of a product follows the rotation, as `quaternion/mtimes` does. `vector3d.py` (`vector3d`, `byPolar`, `X`/`Y`/`Z`,
`rand`, polar coordinates, `normalize`, `dot`, `cross`, `angle`, `norm`, `orth`, `perp`,
`mean`, `unique`, the antipodal semantics); `quaternion.py` (`quaternion`, `rotation`,
`orthogonalTransform`, products, `inv`, `dot`, `angle`, `axis`, `unique`, `mean`, `exp`,
`log`, Haar `rand`, `inversion`, `reflection`); `representations.py` (`byEuler` in the
five conventions with the degenerate case, `Euler` with the convention argument and the
session preference, `byAxisAngle`, `byMatrix`, `matrix`, `byRodrigues`, `Rodrigues`,
`map`, `fit`); `homochoric.py` (`byHomochoric`, `homochoric`, `cubochoric`);
`plottingconvention.py` (the value type, the pictogram grammar, `ij`, `default3D`,
`plotx2east` and friends); `units.py` (`degree`); `display.py` (`__repr__` mirroring
MATLAB's display). Tests: `test_semantics.py` S1 to S18, `test_vector3d.py`
(`check_vector3d`), `test_rotation.py` (`check_eulerquat`, `unique`, `mean`, float32,
exp/log, the small-angle cases of the Julia audit), `test_homochoric.py`
(`check_homochoric`), `test_plottingconvention.py`.

**G2 - point groups. DONE 2026-09-21** (`latticetype.py`, `symmetry.py`, `data/pointgroupnames.py`; the closure grows by products with the generators and compares elements pairwise, so the icosahedral group builds in a tenth of a second; `symmetry(rot)` identifies a group from its elements by signature then by elements; `tests/test_symmetry.py` with the Crystalline fixture as `tests/fixtures/pointgroups.json`; 139 tests). `numSym()` and `numProper()` are methods, as in MATLAB; `isLaue`, `isProper`, `lattice`, `multiplicityZ` are properties. `latticetype.py`, `symmetry.py`: `symmetry('m-3m')`,
`symmetry('321', axes)`, `symmetry(rot)` by closure, the name table
(`data/pointgroupnames.py`, MTEX's order and aliases), `Laue`, `properGroup`,
`properSubGroup`, `disjoint`, `union`, `factor`, `nfold`, `isLaue`, `isProper`,
`numSym`, `numProper`, `lattice`, `LaueName`, `pointGroup`, `id`, `multiplicityZ`,
`multiplicityPerpZ`, identification of a group given by elements. Tests:
`test_symmetry.py` with the Crystalline fixture of the Julia port
(`../mtexJulia/test/fixtures/pointgroups.jl`, converted once to JSON).

**G3 - frames and the session. DONE 2026-09-21** (`crystalaxes.py` as kernels on (3, 3) row arrays, `frames.py` with `referenceFrame`, `crystalFrame`, `specimenFrame`, `notIndexedFrame`, the register, the siblings and the named frames; the session holds the register, the name map and the default frame, and `plottingConvention.default` reads the default frame's convention; 151 tests). Two spellings worth knowing: positional strings in `crystalFrame` are alignments, everything else a keyword; and a frame called with a group name is that frame in that group, so `specimenFrame.rolling('mmm')` reads as in MATLAB. The constructors answer with the registered instance through `__new__`, so `crystalFrame(...) is crystalFrame(...)` holds. The transition matrix is `inv(B) A` on row axes, the rotation between two settings of one lattice. `crystalaxes.py` (one construction for every lattice,
the alignment grammar `X||a`, `Z||c*`, EDAX, `rotAxes`, `reciprocalAxes`, `alignment`);
`frames.py` (`referenceFrame`, `crystalFrame`, `specimenFrame`, `notIndexedFrame`, the
register, the siblings, `specimenFrame.specimen`/`measurement`/`rolling`/`geological`/
`default`, `makeDefault`, `isAligned`, `isCompatible`, `transformationMatrix`, `eqTol`,
`sim`, `axes`, `abc`, `abg`, `mineral`, `color`, `how2plot`, `pfAnnotations`);
`session.py` (register, name map, default frame, preferences, `getMTEXpref`,
`setMTEXpref`, `reset`). Tests: `test_frames.py` (`check_crystalAxes`,
`check_symmetryCompare`, the frame parts of `check_referenceFrame`,
`check_plottingConventionOwnership`), `test_session.py`.

**G4 - framed vectors and orientations. DONE 2026-09-21** (`vector3d` carries `frame` and `dispStyle`, with the index accessors `hkl`, `hkil`, `uvw`, `UVTW` and their setters, `CS` keeping the indices and `frame` keeping the components, `dspacing`, `round`, `char`; `miller.py` with `Miller` in every constructor form of the branch; `orientation.py` with `frameA`, `frameB`, `antipodal` and the `by*` constructors taking frames positionally; `products.py` with `fitFrame`, `dropGroup` and `mtimes`; `joinFrames` in `frames.py` as the ADR 0009 rule, used by `cat`, by `__setitem__` and by every binary operation; 161 tests). Two decisions taken here: grain exchange between two frames is an error, as in Julia, where MATLAB warns; and `xvector`, `yvector`, `zvector` are minted on every access through a module `__getattr__`, since a shared mutable constant took a frame in one test and carried it into the next. `miller.py` (`Miller` in every constructor
form of the branch, the index grammar with three and four indices and the bracket
strings, `hkl`, `hkil`, `uvw`, `UVTW`, `dspacing`, `round`, `dispStyle`, display with the
brackets of the lattice); `orientation.py` (`orientation(rot, cs, ss)`, the `by*`
constructors, `CS`, `SS`, `frameA`, `frameB`, `inv` swapping the frames, `antipodal`,
`isMisorientation`, `rand`, display with the Euler table). Tests: `test_miller.py`
(`check_Miller`), `test_orientation.py`, the containers part of `test_semantics.py`.

**G5 - symmetry-aware operations. DONE 2026-09-21** (`symreduce.py` with the pair kernel vectorised over broadcast pairs, one matrix product per group element; `symmetryops.py` with `symmetrise`, `multiplicity`, the reduced `dot`, `angle`, `==`, `unique`, `mean`, `axis` and `project2FundamentalRegion` for directions and orientations; `sphericalregion.py`, `orientationregion.py`, `frametransform.py`, `standardorientations.py` with `byMiller`, the relationships, `variants` and `CSL`; every semantics statement S1 to S26 passes; 258 tests). Two traps met on the way: a base-class method that calls `self.dot` is intercepted by the subclass's symmetry-reduced version, so the chord angle reads the coefficients directly; and the orbit images keep the group axis next to the coordinates so that two lists of any shapes broadcast on their leading axes. `symreduce.py` (the one kernel `symReduce(q1, q2,
L, R, exchange)`, `symDots`, `symAngles`, `reductionGroups`); `products.py` (`fitFrame`,
`dropGroup`, `o * o`, `rot * o`, `o * rot`, `o * v`, `rotate`); `symmetryops.py`
(`symmetrise`, `multiplicity`, `dot`, `angle`, `==`, `unique`, `mean`, `find`);
`sphericalregion.py` (`sphericalRegion`, `fundamentalSector` as the Dirichlet domain,
`conventionalSector`, `project2FundamentalRegion` for vectors, `checkInside`, `vertices`,
`boundary`, `thetaIntervals`); `orientationregion.py` (`orientationRegion`,
`fundamentalRegion`, `project2FundamentalRegion`, `maxAngle`, `checkInside`);
`frametransform.py` (`align`, `byScreenAlignment`, `transformReferenceFrame`);
`standardorientations.py` (`byMiller`, `cube`, `goss`, `brass`, `copper` and the rest,
the named orientation relationships, `variants`). Tests: `test_symmetryops.py`,
`test_products.py`, `test_sphericalregion.py` (`check_fundamentalRegion`,
`check_antipodalDistance`), `test_orientationregion.py`, `test_standardorientations.py`.

**G6 - grids. DONE 2026-09-21** (`grids/s1grid.py`, `s2grid.py` with `S2Grid` as a `vector3d` subclass that demotes to a plain list under any operation that breaks its rings, `equispacedS2Grid` in the MATLAB forms plus the frame form, `regularS2Grid`; `fibonacci.py`; `so3grid.py` with the coset-symmetrised search ported from the Julia port; `localgrids.py`; `grids/search.py` with `find` on plain lists in the three MATLAB forms, `check_find` ported; 290 tests). `find` returns the MATLAB pair: the index with the angle, the k columns, or two sparse matrices, incidence and angle. The orientation grid search is a Python loop per query and per coset representative, about ten milliseconds a query; a vectorised form is the first thing P2 or P3 will ask for. One NumPy trap: an ndarray has a `.data` attribute too, so a `hasattr` check for a container reads the raw buffer; the code tests `isinstance(x, MtexArray)`. `grids/s1grid.py`, `s2grid.py` (`S2Grid` with the ring structure,
`find`, `equispacedS2Grid`, `regularS2Grid`, `subGrid`), `fibonacci.py`, `so3grid.py`
(`SO3Grid`, `equispacedSO3Grid`, `find`, the coset-symmetrised search), `localgrids.py`.
Tests: `test_s2grid.py`, `test_so3grid.py`, `test_localgrids.py`, `test_fibonacci.py`
with the Julia assertions (search against a full scan, coverage, spacing, uniqueness),
`check_find`.

**G7 - the rest.** `tangent.py` DONE 2026-09-21 (`SO3TangentVector` as a `vector3d` subclass carrying `side` and `oriRef`, the frame read through to the reference; `log` reduces the pair first so its norm is `angle`; `exp` rewraps with the frames of the reference instead of going through the product rules, which would strip a group; a rotation on the left moves a left vector with its reference, on the right a right vector, as the Julia port has it and the branch does not; what is no longer a tangent vector demotes to a plain framed vector; check_logReference ported in full, the pair carriage of check_SO3TangentVectors, the identities of the Julia page; 330 tests). One trap: a `(3, 3)` coordinate array of three vectors reads as a skew matrix, so the internal constructors hand in a `vector3d`. The import cost above NumPy is 200 ms since G6, from `scipy.sparse` imported at module level in the grids, not under 10 ms as the status says. `fibre.py` DONE 2026-09-21 (`fibre` holding `o1`, `o2` and `h`, `r` and the frames derived, a list when the parts have a shape; `angle(f, ori)` as the Julia orbit formula over both groups from the chord of the closest pair; `symmetrise`, `multiplicity` with `(h, r)` and `(-h, -r)` counted once; `==` up to the symmetry of the frames as MATLAB has it; the named rolling fibres with the far end projected next to the near one; `fibre.fit` as the local quaternion tensor fit with `diagnostics` for the MATLAB triple and `noSymmetry` for the representatives as stored, the pole figure search waiting for P2; 337 tests). One lesson: a cloud gathered at its symmetry-aware mean is only consistent when it fits the fundamental region of both groups, and with a 222 specimen group on top of 432 a 30 degree arc does not. `hull.py` DONE 2026-09-21 with P3's M6: `convexHull` in the plane, Andrew's chain as a numba loop with the Python loop as oracle; the hull of points in space and the spherical Voronoi diagram join it with P2's pole figures. `slipsystem.py` DONE 2026-09-21 with P3's M7: `slipSystem` and `dislocationSystem` over arrays, the family table, `symmetrise`, `SchmidFactor`, the dyads, `ori * sS`. `crystalshape.py` DONE 2026-09-22 with `data/crystalshapes.py` (`crystalShape(N, habitus, extension)` with MATLAB's properties `N`, `V`, `F`, `E`, `CS`, `faceArea`, `faceCenter`, `volume`, `diameter` and Julia's representation: the symmetrised distinct normals bound the half spaces, the vertices are the triple plane intersections inside all of them as one batched solve over the triples, the faces convex polygons ordered around their normals, normals without a face dropped, the whole scaled to a diameter of one; `N` keeps the forms as given in the crystal frame and `V` carries the frame the shape stands in, so `ori * cS` lands in the specimen and `cS.CS` stays the crystal; a list of shapes is one object over vertices of shape `(n, nV)` sharing `F`, made by `ori * cS`, `s * cS`, `cS + v` with one orientation, factor or shift per shape, and an object with its own `rotate` takes `v + cS` over from `vector3d`; the face normals are derived from the ordered polygons, so they follow every move; `cS[Miller]` selects faces within a degree of the given normals in crystal coordinates, `innerFace` the section through the centre, `exitDistance` how far a direction runs inside; `crystalShape.cube/hex/tetr(cs)` and the 48 minerals of the MATLAB catalogue as static constructors, `crystalShape.quartz(cs, simple=True)`, built on a given frame of the catalogue's Laue class; `fitArea` is not ported, MATLAB has it as a stub; 5 tests, the Julia numbers and the counts of the MATLAB CrystalShapes page). `plane.py` DONE 2026-09-22 (`plane3d(N, d)`, `plane3d(N, v0)`, `plane3d.byPoints(v)` as Julia's least squares fit through the centroid rather than MATLAB's `perp` with the first point; the unit normal carries the frame, `d` an array, a list of planes over a list of normals; `dist` to points and between parallel planes (NaN where they meet), `project`, `intersect(v1, v2)` with NaN outside the segment and `lineIntersect`, `clip(lo, hi)` the polygon cut from a box in order around the normal, `rot * plane`; `plot(plane)` in `plotting/shapeplot.py` clips the plane by the bounds of the 3D axes as MATLAB does; 3 tests, the Julia plane.jl numbers). `parent2child.py` DONE 2026-09-22 (`calcParent2Child` as ADR 0005's trimmed chordal misfit with the fixed point iteration, backtracking, the deterministic angle-strided subsamples and the grid scan, `local=True`, `fit=True` for MATLAB's second output, `misfit(mori, p2c)`; the misfit and the scan vectorised as outer angle tables in blocks; `calcVariantId` returning the named tuple `(variantId, packetId, bainId, fit)` one based as MATLAB counts, with the symmetric dot for the packet and Bain families; `calcParent` in MATLAB's four settings, the child with parent form reducing over the parent group alone so that `child * PV[id]` is the parent as stored, pairs and triples for M9's votes, many children of one parent by the weighted robust mean; `round2Miller` as MATLAB's penalised search vectorised over the coprime index table, the four winners with their group stripped and the four index styles kept; 3 tests, MATLAB's check_calcParent2Child and the Julia numbers). G7 is complete. (`calcParent2Child`,
`calcVariantId`, `calcParent`, `round2Miller`). Tests: `test_tangent.py`
(`check_logReference`, `check_SO3TangentVectors`), `test_fibre.py`, `test_slipsystem.py`,
`test_crystalshape.py`, `test_parent2child.py` (`check_calcParent2Child`).

Each step ends with the import-time measurement of E4 and the test count.

## 6. Source layout

```
mtex/geometry/
  _array.py            the container base: shape, indexing, views, cat, broadcasting
  units.py             degree
  display.py           __repr__ of every class, MATLAB's layout
  vector3d.py          vector3d
  quaternion.py        quaternion, rotation, orthogonalTransform, the kernels
  representations.py   Euler conventions, axis-angle, matrix, Rodrigues, map, fit
  homochoric.py
  plottingconvention.py
  latticetype.py
  symmetry.py          the point group
  data/pointgroupnames.py, data/crystalshapes.py
  crystalaxes.py
  frames.py            referenceFrame, crystalFrame, specimenFrame, notIndexedFrame, register
  session.py
  miller.py            Miller and the index grammar
  orientation.py
  symreduce.py
  products.py
  symmetryops.py
  sphericalregion.py
  orientationregion.py
  frametransform.py
  standardorientations.py
  grids/s1grid.py, s2grid.py, fibonacci.py, so3grid.py, localgrids.py
  hull.py, tangent.py, fibre.py, plane.py, slipsystem.py, crystalshape.py, parent2child.py
tests/test_<file>.py   one per source file, plus test_semantics.py and test_oracles.py
```

File names are the Julia ones so the two ports read side by side; class names inside are
MATLAB's. A file imports only from files above it in this list; `tests/test_layers.py`
checks it.

## 7. Experiments this subplan depends on

- E1 of `docs/PLAN.md`, run in G1 on `vector3d` and `rotation` and again in G5 on the
  symmetry reduction: `(..., 4)` against four columns, float64 against float32, one
  `einsum` over the group against a loop over elements. Budget: the Julia kernel at
  46 ns per element single-threaded, MATLAB at whatever the bridge reports for
  `angle(ori, ori0)` on 1e6 cubic orientations. Run 2026-09-21 in Subplan B
  (`docs/research/benchmark-geometry.md`): `dot(ori, ref)` at 382 ns against Julia's 49
  and MATLAB's 115 with NumPy, 61 with the numba kernels of `kernels.py` taken the same
  day (D5). `symmetryops.py` took the list-of-references form of
  `project2FundamentalRegion` for the benchmark. The nearest search of `S2Grid` and
  `SO3Grid` was compiled the same day (`docs/research/benchmark-gridsearch.md`,
  `grids/gridsearch.py`): the structured ring search as one parallel numba loop, 730x and
  1000x the Python search. Then reconsidered for ODF reconstruction
  (`docs/research/benchmark-gridsearch.md`): the search that matters there is the radius
  search, so the SO(3) grids now search through a quaternion k-d tree
  (`grids/orientationtree.py`, nearest + k-nearest + radius, exact), the bespoke ring
  kernel is retired for SO(3), and `superfibonacciSO3Grid` was added as a low-discrepancy
  alternative grid. The S2 ring grid keeps its compiled search.
- E4, import time, at the end of every step. `import mtex.geometry` must stay under the
  NumPy and SciPy import cost plus 50 ms.
- E5, float32, in G5: the dot error of 2e-7 caps the angle resolution near 1e-3 rad, as
  the Julia benchmark found; the subplan records where float32 is accepted.

## 8. Tests

- Every `check_*` named in section 5 becomes assertions in the test of the file that
  owns the subsystem, same tolerances; a check that cannot be ported yet is listed in
  `tests/TODO.md` with the reason.
- `test_semantics.py` holds the G0 statements; `test_oracles.py` the library
  comparisons; orix and SciPy are test dependencies only.
- The point group fixture is frozen from the Julia fixture; neither Julia nor Crystalline
  is a test dependency.
- Numbers pinned from MATLAB, through the bridge on request only: the crystal axes per
  lattice and alignment, the fundamental sector normals per point group, the Euler
  angles of the standard orientations. Numbers pinned from Julia by running
  `../mtexJulia` freely.
- A fixture resets the session before every test.

## 9. Left out of P1

Plot methods (P5); `spatialTransforms` (P3 with EBSD); `calcDensity`, `calcODF`,
`calcKernel`, `calcBinghamODF`, `calcFourierODF`, `calcMIndex`, `calcTensor`,
`volume`, `fibreVolume` and everything returning a function or a tensor (P2, P4);
`embedding` (P4); the quadrature grids and the homochoric grid (P2/P3 with their
consumers); `S2Triangulation`, `calcDelaunay`, `calcVoronoi` (P2/P3); `gridLayout`, the
CSL table, hierarchical clustering, `calcCluster` (their consumers); `mtexdata`;
HEALPix and `DelaunaySO3` (dropped, as in Julia).

## 10. Open items to settle during G0 to G4

- The name of the O(3) class: `orthogonalTransform` as in Julia, or `rotoinversion`.
- Whether `vector3d.X` is a class attribute (needs a class-level property descriptor) or
  only the module constants `xvector`, `yvector`, `zvector`; proposal: both.
- Whether `angle(o1, o2)` on two arrays is elementwise with broadcasting (proposal, as
  MATLAB) and `angle_outer` the outer form.
- Whether `Miller` returns a `vector3d` (as the branch) or a `Miller` subclass that
  exists only for `isinstance` and display; proposal: `vector3d`, following the branch,
  since the frame already says everything.
- Setter semantics of section 3, mutate versus raise.
- Which builtin and NumPy names the `mtex` namespace shadows under `from mtex import *`
  (`abs`, `round`, `sum`, `mean`, `dot`, `cross`, `norm`, `unique`, `angle`, `min`, `max`);
  proposal: the MTEX functions accept plain arrays and forward to NumPy, so shadowing
  costs nothing.
