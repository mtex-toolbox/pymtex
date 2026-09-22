// the two kernels of bench/e2_kernels.py in C
#include <math.h>
// Hamilton product on (n,4) rows
void qmul(const double *p, const double *q, double *out, long n) {
  for (long i = 0; i < n; i++) {
    const double *a = p + 4*i, *b = q + 4*i; double *o = out + 4*i;
    o[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    o[1] = a[1]*b[0] + a[0]*b[1] - a[3]*b[2] + a[2]*b[3];
    o[2] = a[2]*b[0] + a[3]*b[1] + a[0]*b[2] - a[1]*b[3];
    o[3] = a[3]*b[0] - a[2]*b[1] + a[1]*b[2] + a[0]*b[3];
  }
}
// best |<q, e>| over m images for each of n rows
void bestdot(const double *qs, const double *es, long n, long m, double *val, long *idx) {
  for (long i = 0; i < n; i++) {
    const double *q = qs + 4*i; double best = -1; long k = 0;
    for (long j = 0; j < m; j++) {
      const double *e = es + 4*j;
      double d = fabs(q[0]*e[0] + q[1]*e[1] + q[2]*e[2] + q[3]*e[3]);
      if (d > best) { best = d; k = j; }
    }
    val[i] = best; idx[i] = k;
  }
}
