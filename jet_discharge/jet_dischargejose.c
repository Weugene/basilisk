#include "axi.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

#define Ln 2. // Lenght of the tube
#define We 1.
#define Re 1000.

int LEVEL = 11;
double uemax = 0.01;

u.n[left]  = dirichlet(2.*(1. - sq(y)) * (sq(y) < 1));
u.n[right] = neumann(0);
p[right]   = dirichlet(0);
f[left]    = (y < 1);

bid needle;
u.t[needle] = dirichlet(0);
u.n[needle] = dirichlet(0);
FILE * fp;

int main (){
  fp = popen("ppm2mp4 movie.mp4", "w");
  L0 = 64.;
  init_grid(64);
  rho1 = 1., rho2 = 0.001;
  mu1 = 1./Re, mu2 = 0.01/Re;
  f.sigma = 1/We;

//  f[needle] = 0;
//  run();
  f[needle] = neumann(0);
  run();
  f[needle] = 1;
  run();
}

event init (t = 0) {
  mask(x < 2 && y > 1 ? needle : none);
  refine(level < LEVEL && x < 4 && y < 3);
  fraction (f, difference (Ln/2. - x, y - 1.));
  foreach()
    u.x[] = f[]*2.*(1.-sq(y));
  boundary ({f, u.x});
}

event adapt (i++)
  adapt_wavelet ({f, u.x, u.y}, (double[]){0.01, uemax, uemax}, LEVEL);

//event location (t += 0.1) {
//  scalar pos[];
//  position(f, pos, {1, 0});
//  fprintf(stdout,"%g %g \n", t, statsf(pos).max);
//  fflush(stdout);
//}

event mov (t += 0.025; t <= 5.){
  output_ppm(f, fp = fp, n = 1024, box = {{X0, Y0},{10, 5}});
}

int iteration = 0;
//Output
#include "output_fields/output_vtu_foreach.h"
event vtk_file (t += 0.1; t<=30)
{
  int nf = iteration;
  scalar l[];
  foreach()
  l[] = level;

  char name[80], subname[80];
  FILE *fp;
  sprintf(name, "hs_%4.4d_n%3.3d.vtu", nf, pid());
  fp = fopen(name, "w");
  output_vtu_bin_foreach((scalar *) {l, f, p}, (vector *) {u}, 64, fp, false);
  fclose(fp);
  sprintf(name, "hs_%4.4d.pvtu", nf);
  sprintf(subname, "hs_%4.4d", nf);
  fp = fopen(name, "w");
  output_pvtu_bin((scalar *) {l, f, p}, (vector *) {u}, 64, fp, subname);
  fclose(fp);
  iteration++;
}
