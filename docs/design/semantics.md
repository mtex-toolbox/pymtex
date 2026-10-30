# Semantics of rotations, orientations and frames

Status: G0 of the geometry subplan, 2026-09-21. The statements are those of
`../mtexJulia/docs/design/semantics.md`, respelled in the syntax of the frameSymmetry
branch. Every numbered statement has a test of the same number in
`tests/test_semantics.py`; a test whose classes are not built yet is skipped, not
missing. Sources: `doc/Rotations/RotationDefinition.m`,
`doc/CrystalOrientations/DefinitionAsCoordinateTransform.m`,
`doc/CrystalOrientations/MTEXvsBungeConvention.m`, `geometry_tools/euler2quat.m`,
`@quaternion/Euler.m`, `@vector3d/rotate.m`, `@orientation/times.m` and `dot.m` of the
branch.

## Rotations

**S1. A rotation acts actively.** `r * v` moves the direction `v`; the frame stays.
`rotation.byAxisAngle(vector3d.Z, 90*degree) * vector3d.X == vector3d.Y`.

**S2. Products compose right to left.** `(r1 * r2) * v == r1 * (r2 * v)`; the right
factor acts first, as for matrices.

**S3. Inverse.** `inv(r) * (r * v) == v`; `inv(r)` is the conjugate quaternion.

**S4. Matrix.** `matrix(r) @ v.xyz == (r * v).xyz`, `matrix(r)` is orthogonal with
determinant `+1`, and `rotation.byMatrix(matrix(r)) == r`. `rotation.byMatrix` of an
orthogonal matrix with determinant `-1` raises; `orthogonalTransform.byMatrix` accepts
both signs. A rotation is never silently improper.

**S5. Equal up to sign.** `rotation(a, b, c, d) == rotation(-a, -b, -c, -d)`; a
`quaternion` is not equal to its negative. The stored sign is preserved by every
operation that returns a `rotation`, never canonicalised. `angle(r) == 2*acos(|a|)` lies
in `[0, pi]`, and is computed from `atan2(|(b, c, d)|, |a|)` so that it is accurate near
`0` and `pi`.

**S6. Axis and angle.** `rotation.byAxisAngle(n, omega)` rotates by `omega` about `n` in
the right-hand sense; `axis(r)` is a normalised `vector3d`, chosen so that
`angle(r) <= pi`, and `vector3d.X` for the identity;
`rotation.byAxisAngle(axis(r), angle(r)) == r`. What MATLAB implements as a method is a
Python method and a function, `r.axis()` and `axis(r)`; what MATLAB declares as a
property is a Python property, `r.phi1`, `v.isNormalized`, `o.isImproper`.

**S7. Rodrigues vector.** `Rodrigues(r) == tan(omega/2) * axis(r)`;
`rotation.byRodrigues(Rodrigues(r)) == r` for `omega < pi`.

**S8. Two directions fix a rotation.** `rotation.map(u, v)` is the smallest rotation
with `r * u` parallel to `v`; `rotation.map(u1, v1, u2, v2)` maps `u1` onto `v1` and `u2`
onto `v2` and requires `angle(u1, u2) == angle(v1, v2)`.

## Euler angles

All conventions are stated as active products of elementary rotations about the fixed
axes, right factor first. With `Rz`, `Ry`, `Rx` the elementary rotations:

**S9. Matthies, ZYZ.** `rotation.byEuler(alpha, beta, gamma, 'ZYZ') == Rz(alpha) *
Ry(beta) * Rz(gamma)`. Its quaternion is `euler2quat`'s: `(cos(beta/2) cos((alpha +
gamma)/2), -sin(beta/2) sin((alpha - gamma)/2), sin(beta/2) cos((alpha - gamma)/2),
cos(beta/2) sin((alpha + gamma)/2))`. `'Matthies'` names the same convention.

**S10. Bunge, ZXZ.** `rotation.byEuler(phi1, Phi, phi2, 'Bunge') == Rz(phi1) * Rx(Phi) *
Rz(phi2)`, equal to `rotation.byEuler(phi1 - pi/2, Phi, phi2 + pi/2, 'ZYZ')`. This is
SciPy's intrinsic `Rotation.from_euler('ZXZ', [phi1, Phi, phi2])`. Bunge is the default
convention: `rotation.byEuler(phi1, Phi, phi2)` reads
`getMTEXpref('EulerAngleConvention')`, which a fresh session sets to `'Bunge'`.

**S11. Roe.** `rotation.byEuler(Psi, Theta, Phi, 'Roe') == rotation.byEuler(Psi, Theta,
Phi, 'ZYZ')`, the same rotation under other names.

**S12. Kocks.** `rotation.byEuler(Psi, Theta, phi, 'Kocks') == rotation.byEuler(Psi,
Theta, pi - phi, 'ZYZ')`.

**S13. Canova.** `rotation.byEuler(omega, Theta, phi, 'Canova') == rotation.byEuler(pi/2
- omega, Theta, 3*pi/2 - phi, 'ZYZ')`.

**S14. Extraction and the degenerate case.** `Euler(rotation.byEuler(a, b, c))` returns
a triple that rebuilds the same rotation for every input; the first and third angle are
returned in `[0, 2*pi)`, the second in `[0, pi]`. When the middle angle is zero only the
sum of the outer two is determined: the extraction returns the whole rotation in the
first angle and `0` in the third. `Euler(r, 'ZYZ')` extracts in that convention;
`r.phi1`, `r.Phi`, `r.phi2` are the Bunge angles.

**S15. Degrees are explicit.** Angles are radians; `30*degree` is `30*pi/180`. Nothing
converts on the user's behalf and no heuristic warns about large angles.

## Improper elements

**S16. `orthogonalTransform`.** `-r` is `r` followed by the inversion;
`rotation.inversion` maps every `v` to `-v`; `reflection(n)` fixes every `v`
perpendicular to `n` and sends `n` to `-n`, and equals `-rotation.byAxisAngle(n, pi)`.
Parity composes by exclusive or. `matrix` of an improper element has determinant `-1`.
`rotation` has no parity: `isImproper` exists on `orthogonalTransform` only.

## Vectors

**S17. Normalised and antipodal vectors.** `normalize(v)` returns a vector with
`isNormalized == True`, which `theta`, `rho`, `angle` and the projections trust. For
`a = vector3d(x, y, z, antipodal=True)`, `angle` between two such is `<= pi/2` and
`a == vector3d(-x, -y, -z, antipodal=True)`; a comparison with one antipodal operand is
antipodal. `a + a` is a plain `vector3d`: arithmetic clears both flags. `r * v` keeps
both flags: a rotated direction is a direction, a rotated axis an axis.

**S18. Polar coordinates.** `v.theta` is the angle to `vector3d.Z` in `[0, pi]`,
`v.rho` the azimuth from `vector3d.X` towards `vector3d.Y` in `[0, 2*pi)`, as MATLAB's
`polar` returns it; `vector3d.byPolar(theta, rho)` inverts them and is normalised.

## Orientations and frames (tested from G4/G5 on)

**S19. An orientation maps crystal coordinates to specimen coordinates.** `ori * h`
takes the coordinates of a crystal direction `h` in `ori.frameA` to coordinates in
`ori.frameB`. `rotate(h, ori)` is the same operation. `ori.CS` is `ori.frameA` and
`ori.SS` is `ori.frameB`, positionally: `inv(ori).CS` is the specimen side. Bunge's map
is `inv(ori)`; MTEX's `matrix(ori)` is the transpose of Bunge's `g`.

**S20. Left is specimen, right is crystal.** `rot * ori` turns the specimen: every
crystal direction moves with it, `(rot * ori) * h == rot * (ori * h)`; `ori * rot` turns
the direction inside the lattice first. A rotation that is not an element of the group
on that side drops the group from that side's frame, silently, and keeps the frame.

**S21. Products of orientations.** `o1 * o2` requires `o2.frameB` to fit `o1.frameA`
(the same handle, or compatible crystal frames, whose transition is absorbed) and
yields the map from `o2.frameA` to `o1.frameB`. `inv(ori)` swaps the frames.

**S22. Symmetry acts on the right for the crystal, on the left for the specimen.**
`s_s * ori * s_c` for `s_s` in the specimen group and `s_c` in the crystal group
describes the same physical orientation. `symmetrise(ori)` lists the proper equivalents
only, `ori.CS.numProper * ori.SS.numProper` of them.

**S23. Misorientation.** `mori = inv(o1) * o2` maps `o2.frameA` to `o1.frameA`;
`angle(o1, o2)` is the minimum of `angle(inv(o1) * o2)` over all symmetric equivalents
of both, and with `mori.antipodal` (grain exchange, allowed only when `frameA is
frameB`) also over `inv(mori)`.

**S24. A crystal direction reduces over the group of its frame.** `dot(m1, m2)`,
`angle`, `==`, `unique` compare `m1` against every `s * m2` for `s` in the group of the
frame, unless `symmetry=False`. A specimen frame's group does not enter vector
comparison.

**S25. Frame change is not rotation.** `transformReferenceFrame(x, fr)` re-expresses `x`
in `fr` and moves nothing; `rotate(x, r)` moves `x` and keeps its frame; rotating by an
orientation lands in `ori.frameB`.

**S26. The default convention.** A fresh session's default specimen frame is
`specimenFrame.specimen`, the generic `X, Y, Z` frame, drawn with `x` east, `y` south
and `z` into the screen (`plottingConvention.ij`), the convention of SEM images. A
frame without a convention follows the session default at render time.

**S27. A fibre is that of every equivalent pair.** `fibre(h, r)` stands for the orientations
mapping an image of `h` under the crystal point group onto an image of `r` under the
specimen point group, the improper elements included, as S24 compares directions:
`angle(f, ori)`, `volume(ori, f, radius)`, `fibreVolume` and `symmetrise(f)` all take that
union. Where either group holds the inversion it holds the fibre of `-h || r`, which no
rotation of the symmetry reaches: in `m-3m` the orientation turning `-(124)` onto `z` lies
on the fibre `(124) || z`.
