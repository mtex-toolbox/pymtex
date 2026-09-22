// the local Voronoi clipping of mtex/maps/kernels.py as a plain C loop, for the E2
// comparison of numba against a compiled extension. Same algorithm, same candidate
// lists, one thread; returns the number of faces (site < neighbour) written.
//
//   gcc -O2 -shared -fPIC -o decomposition_kernel.so decomposition_kernel.c
#include <math.h>
#include <stdint.h>
#include <string.h>

#define K 256
#define NO_SITE -1

static int clip(double *vs, int32_t *gs, int m, double *ws, int32_t *hs,
                double px, double py, double qx, double qy, int32_t gen) {
  double nx = qx - px, ny = qy - py;
  double c = (nx * qx + ny * qy + nx * px + ny * py) / 2;
  int cnt = 0;
  for (int i = 0; i < m; i++) {
    int i2 = (i == m - 1) ? 0 : i + 1;
    double v1x = vs[2 * i], v1y = vs[2 * i + 1], v2x = vs[2 * i2], v2y = vs[2 * i2 + 1];
    double d1 = nx * v1x + ny * v1y - c, d2 = nx * v2x + ny * v2y - c;
    if (d1 <= 0) {
      if (cnt + 2 > K) return -1;
      ws[2 * cnt] = v1x; ws[2 * cnt + 1] = v1y; hs[cnt] = gs[i]; cnt++;
      if (d2 > 0) {
        double t = d1 / (d1 - d2);
        ws[2 * cnt] = v1x + t * (v2x - v1x); ws[2 * cnt + 1] = v1y + t * (v2y - v1y); hs[cnt] = gen; cnt++;
      }
    } else if (d2 <= 0) {
      if (cnt + 1 > K) return -1;
      double t = d1 / (d1 - d2);
      ws[2 * cnt] = v1x + t * (v2x - v1x); ws[2 * cnt + 1] = v1y + t * (v2y - v1y); hs[cnt] = gs[i]; cnt++;
    }
  }
  memcpy(vs, ws, 2 * cnt * sizeof(double));
  memcpy(gs, hs, cnt * sizeof(int32_t));
  return cnt;
}

// the cell of site s against one candidate list; *open says whether an edge of the seed
// box survived; returns the vertex count or -1 on overflow
static int cellOf(double *vs, int32_t *gs, double *ws, int32_t *hs, const double *pos, const int32_t *ij,
                  const int32_t *lookup, int64_t n1, int64_t n2, int64_t off0, int64_t off1, int64_t s,
                  const int64_t *offs, const double *dIdeal, int64_t nOffs, double W, double distortTol, int *open) {
  double px = pos[2 * s], py = pos[2 * s + 1];
  int64_t a0 = ij[2 * s] - off0, b0 = ij[2 * s + 1] - off1;
  vs[0] = px - W; vs[1] = py - W; vs[2] = px + W; vs[3] = py - W;
  vs[4] = px + W; vs[5] = py + W; vs[6] = px - W; vs[7] = py + W;
  for (int i = 0; i < 4; i++) gs[i] = NO_SITE;
  int m = 4;
  double rmax = W * sqrt(2.0);
  int stillOpen = 1;
  for (int64_t k = 0; k < nOffs; k++) {
    if (!stillOpen && dIdeal[k] * (1 - distortTol) > 2 * rmax) break;
    int64_t a = a0 + offs[2 * k], b = b0 + offs[2 * k + 1];
    if (a < 0 || a >= n1 || b < 0 || b >= n2) continue;
    int32_t s2 = lookup[a * n2 + b];
    if (s2 < 0) continue;
    m = clip(vs, gs, m, ws, hs, px, py, pos[2 * s2], pos[2 * s2 + 1], s2);
    if (m < 0) return -1;
    rmax = 0; stillOpen = 0;
    for (int i = 0; i < m; i++) {
      double dx = vs[2 * i] - px, dy = vs[2 * i + 1] - py, r = sqrt(dx * dx + dy * dy);
      if (r > rmax) rmax = r;
      if (gs[i] == NO_SITE) stillOpen = 1;
    }
  }
  *open = stillOpen;
  return m;
}

int64_t localVoronoi(const double *pos, const int32_t *ij, const int32_t *pixel, const uint8_t *isHole,
                     const int32_t *lookup, int64_t n1, int64_t n2, int64_t off0, int64_t off1, int64_t ns,
                     const int64_t *offs1, const double *d1, int64_t m1, double W1,
                     const int64_t *offs2, const double *d2, int64_t m2, double W2,
                     const int64_t *offs3, const double *d3, int64_t m3, double W3,
                     double distortTol, int32_t *cells, double *verts, int64_t cap, int64_t *unbounded) {
  double vs[2 * K], ws[2 * K];
  int32_t gs[K], hs[K];
  int64_t cnt = 0, unb = 0;
  for (int64_t s = 0; s < ns; s++) {
    int isMap = pixel[s] >= 0 || isHole[s];
    int m = 0, open = 1;
    for (int lv = 0; lv < 3; lv++) {
      if (!(open && (lv == 0 || isMap))) break;
      const int64_t *o = lv == 0 ? offs1 : lv == 1 ? offs2 : offs3;
      const double *d = lv == 0 ? d1 : lv == 1 ? d2 : d3;
      int64_t mm = lv == 0 ? m1 : lv == 1 ? m2 : m3;
      double W = lv == 0 ? W1 : lv == 1 ? W2 : W3;
      m = cellOf(vs, gs, ws, hs, pos, ij, lookup, n1, n2, off0, off1, s, o, d, mm, W, distortTol, &open);
      if (m < 0) return -1;
    }
    for (int i = 0; i < m; i++) {
      int32_t g = gs[i];
      if (g == NO_SITE) { if (isMap) unb++; continue; }
      if (g > s) {
        if (cnt >= cap) return -2;
        int i2 = (i == m - 1) ? 0 : i + 1;
        cells[2 * cnt] = (int32_t)s; cells[2 * cnt + 1] = g;
        verts[4 * cnt] = vs[2 * i]; verts[4 * cnt + 1] = vs[2 * i + 1];
        verts[4 * cnt + 2] = vs[2 * i2]; verts[4 * cnt + 3] = vs[2 * i2 + 1];
        cnt++;
      }
    }
  }
  *unbounded = unb;
  return cnt;
}
