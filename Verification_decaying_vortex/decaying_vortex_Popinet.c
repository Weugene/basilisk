/**
# Decaying vortex problem
To verify the spatial accuracy of the present method, the decaying vortex problem is chosen because it is an unsteady problem with an analytical solution:
 $$u(x, y, t) = − \cos 􏱲x \sin 􏱲y \exp{−2􏱲\pi^2 t/Re}$$
 $$v(x, y, t) =   \sin 􏱲x \cos 􏱲y \exp{−2􏱲\pi^2 t/Re}$$
 $$p(x, y, t) =  −\frac14(\cos 2􏱲x + \cos 2􏱲y)\exp{−4\pi^2 t/Re}$$

The computational domain is −1.5 < x, y 􏰒< 1.5 and the IB is located at x = ± 1 and y = ± 1.
The Reynolds number based on the maximum velocity and vortex size is set to 30, and the initial and boundary conditions are given by the analytical solution above.
Simulations are performed till $t=0.3$.
We use the centered Navier-Stokes solver, with embedded boundaries and
advect the passive tracer *f*. */

#define FILTERED
#define RELATIVE_RESIDUAL
#define EPS_MAXA 2
scalar divutmp[];
#include "navier-stokes/centered.h"
#include "../src_local/output_vtu_foreach.h"
#include "../src_local/utils-weugene.h"
#include "fractions.h"

face vector muv[], fs[];
vector uexact[];
scalar pexact[];
scalar omega[];
scalar cs[];
int maxlevel = 8;
int minlevel = 4;
double Ldomain = 3.0, RE = 30.;
double ueps = 1e-5, cseps = 1e-5, peps=1e-3;

void frame_Popinet(scalar cs, face vector fs, double t){
    vertex scalar phi[];
    foreach_vertex() {
        phi[] = ( fabs(x) <= 1 && fabs(y) <= 1 ) ? 1 : -1;
    }
    boundary ({phi});
    fractions (phi, cs, fs);
}

#define uxe (-cos(pi*x) * sin(pi*y) * exp(-2.0 * sq(pi) * t / RE))
#define uye ( sin(pi*x) * cos(pi*y) * exp(-2.0 * sq(pi) * t / RE))
#define pe (-0.25 *  (cos(2*pi*x) + cos(2.0*pi*y)) * exp(-4.0 * sq(pi) * t / RE))
void theory(vector u, scalar p, double t, double RE){
    foreach(){
        u.x[] = uxe;
        u.y[] = uye;
        p[]   = pe;
    }
    boundary({u, p});
    fprintf(ferr, "theory  t= %g\n", t);
}
//$$u(x, y, t) = − \cos 􏱲x \sin 􏱲y \exp{−2􏱲\pi^2 t/Re}$$
//$$v(x, y, t) =   \sin 􏱲x \cos 􏱲y \exp{−2􏱲\pi^2 t/Re}$$
//$$p(x, y, t) =  −\frac14(\cos 2􏱲x + \cos 2􏱲y)\exp{−4\pi^2 t/Re}$$
//in 2D each cells must have 2 BC, in 3D - 3BC
u.t[left]  = neumann(0);
p[left]    = neumann(0);
pf[left]   = neumann(0);

u.t[right] = neumann(0);
p[right]   = neumann(0);
pf[right]  = neumann(0);

u.t[top] = neumann(0);
p[top]   = neumann(0);
pf[top]  = neumann(0);

u.t[bottom] = neumann(0);
p[bottom]   = neumann(0);
pf[bottom]  = neumann(0);

int main(int argc, char * argv[]) {
    if (argc > 1) {
        maxlevel = atoi(argv[1]); //convert from string to float
    }
    size (Ldomain);
    origin (-0.5*Ldomain, -0.5*Ldomain);
//    DT = 1e-8;
    DT = 1e-5;
    CFL = 0.4;
    TOLERANCE = 1e-8;
    RELATIVE_RES_TOLERANCE = 0.1;
    NITERMAX = 30;
    mu = muv;

    N = 1<<maxlevel;
    fprintf(ferr, "maxlevel=%d TOL=%g NITERMAX=%d Re=%g Rel_res_tol=%g\n", maxlevel, TOLERANCE, NITERMAX, RE, RELATIVE_RES_TOLERANCE);
    fprintf(ferr, "CFL=%g cseps=%g peps=%g ueps=%g\n", CFL, cseps, peps, ueps);
    run();
}

/**
We set a constant viscosity corresponding to a Reynolds number of 40, 100,
based on the cylinder diameter (1) and the inflow velocity (1). */

event properties (i++)
{
    foreach_face() muv.x[] = fm.x[]/RE;
}

event wall  (i++)
{
    foreach() {
        u.x[] = cs[]*u.x[] + uxe*(1 - cs[]);
        u.y[] = cs[]*u.y[] + uye*(1 - cs[]);
    }
    boundary ((scalar *){u});
    foreach_face() {
        muv.x[] = fm.x[]/RE;
        uf.x[] = fs.x[]*uf.x[] + uxe*(1 - fs.x[]);
        uf.y[] = fs.y[]*uf.y[] + uye*(1 - fs.y[]);
    }
    boundary ((scalar *){muv});
}

event init (t = 0)
{
    if (!restore (file = "restart")) {
        int it = 0;
        do {
            frame_Popinet (cs, fs, 0);
            theory(uexact, pexact, 0, RE);
        } while ( ++it <= 10 && adapt_wavelet((scalar *){cs, pexact, uexact}, (double []){cseps, peps, ueps, ueps}, maxlevel=maxlevel, minlevel=minlevel).nf != 0);
    }
    foreach() {
        p[] = pexact[];// useless
        foreach_dimension() {u.x[] = uexact.x[];}
    }
    boundary({p, u});
}

event set_dtmax (i++) {
    if (i<=100) {
        NITERMIN=5;
        NITERMAX=100;
    }else{
        NITERMIN=1;
        NITERMAX=100;
    }
    DT *= 1.05;
    DT = min(DT, CFL*Ldomain/pow(2, maxlevel+3));
    fprintf(ferr, "set_dtmax: tnext= %g Dt= %g", tnext, DT);
}

event end_timestep (i++){
    vorticity (u, omega);
    double Luinf = 0, Lpinf = 0, du, dp, maxp = 0, maxu = 0;
    theory(uexact, pexact, t+dt, RE);
    foreach(reduction(max:Luinf) reduction(max:Lpinf) reduction(max:maxu) reduction(max:maxp)){
        if (cs[] == 1){ //in inner frame
            du = sqrt(sq(u.x[] - uexact.x[]) + sq(u.y[] - uexact.y[]));
            dp = fabs(p[] - pexact[]);
            if (du > Luinf) Luinf = du;
            if (dp > Lpinf) Lpinf = dp;
            if (norm(uexact) > maxu) maxu = norm(uexact);
            if (fabs(pexact[]) > maxp) maxp = fabs(pexact[]);
        }
    }
    fprintf (ferr, "i= %d t+dt= %g dt= %g Luinf= %g Lpinf= %g Luinf_rel= %g Lpinf_rel= %g iter_p= %d iter_u= %d \n", i, t+dt, dt, Luinf, Lpinf, Luinf/maxu, Lpinf/maxp, mgp.i, mgu.i);
double eps_arr[] = {1, 1, 1, 1, 1, 1, 1, 1};
MinMaxValues((scalar *){p, pexact, u.x, u.y, uexact.x, uexact.y, omega, divutmp}, eps_arr);
}
/**
We produce animations of the vorticity and tracer fields... */

//event movies (i += 4; t <= 15.)
//{
//    scalar omega[], m[];
//    vorticity (u, omega);
//    foreach()
//            m[] = cs[] - 0.5;
//    boundary ({m});
//    output_ppm (omega, file = "vort.mp4", box = {{-0.5,-0.5},{7.5,0.5}},
//        min = -10, max = 10, linear = true, mask = m);
//#if TURN_ON_TRACER == 1
//    output_ppm (f, file = "f.mp4", box = {{-0.5,-0.5},{7.5,0.5}},
//        linear = false, min = 0, max = 1, mask = m);
//#endif
//}

//Output
//event vtk_file (i++){
event vtk_file (t += 0.01){
    char subname[80]; sprintf(subname, "vortex_Popinet");
    scalar l[]; foreach() l[] = level;
    output_vtu_MPI( (scalar *) {cs, omega, p, pexact, l, divutmp}, (vector *) {u, uexact}, subname, 0 );
}
/**
We adapt according to the error on the embedded geometry, velocity*/
#define ADAPT_SCALARS {cs, omega}
#define ADAPT_EPS_SCALARS {cseps, ueps}
//event adapt (i++){
//    double eps_arr[] = ADAPT_EPS_SCALARS;
//    MinMaxValues(ADAPT_SCALARS, eps_arr);
//    adapt_wavelet ((scalar *) ADAPT_SCALARS, eps_arr, maxlevel = maxlevel, minlevel = minlevel);
//}

event stop (t = 0.3);
/**
## See also

* [Same example with Gerris](http://gerris.dalembert.upmc.fr/gerris/examples/examples/cylinder.html)
*/
