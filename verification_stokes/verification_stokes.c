#define BRINKMAN_PENALIZATION
#define DEBUG_BRINKMAN_PENALIZATION
//#define DEBUG_MINMAXVALUES
//#define DEBUG_OUTPUT_VTU_MPI

scalar fs[];
vector Us[];
face vector n_sol[];
#include "../src_local/centered-weugene.h"

#define MAXLEVEL 10
#define Re 40
#define U_inf (Re/diam)
#define Delta_int (rad/10.)
#undef SEPS
#define SEPS 1e-30

double yo = 0, xo = 0, rad = 0.5, diam = 1;
int iteration = 0;
scalar omega[];
face vector muv[];

u.n[left] = dirichlet(U_inf);
u.t[left] = dirichlet(0);

u.n[right] = neumann(0);
u.t[right] = neumann(0);

u.n[bottom] = neumann(0);
u.t[bottom] = neumann(0);

u.n[top] = neumann(0);
u.t[top] = neumann(0);

p[left]   = neumann(0);
p[right]  = dirichlet(0);
p[top]    = neumann(0);
p[bottom] = neumann(0);

int main() {
    L0 = 10;
    origin(-L0/3, -L0/2);
    eta_s = 1e-6;
    TOLERANCE = 1e-6;
    DT = 1e-4;
    N = 512;
    mu = muv;
    stokes = true;
    run();
}
void normal_calc(scalar f, face vector nf){
    foreach_face() {
        n_sol.x[] = (f[] - f[-1])/Delta;
    }
    
}
event properties (i++) {
    double un;
    scalar magn2[];
    foreach_face() {
        muv.x[] = fm.x[];
        n_sol.x[] = (f[] - f[-1])/Delta;
        magn2[] += sq(n_sol.x[]);
    }

    foreach() {
        theta = atan2(y, x + SEPS);
        coord normal = {cos(theta), sin(theta)};
        un = 0;
        foreach_dimension() un += u.x[] * normal.x;
        foreach_dimension() Us.x[] = u.x[] - un * normal.x;
    }
}

event init (t = 0) {
    int it = 0;
    double theta, r, un;
    do {
        it++;
        foreach() {
            r = sqrt(sq(x) + sq(y));
            theta = atan2(y, x + SEPS);
            coord normal = {cos(theta), sin(theta)};
            fs[] = 0.5*(1 - tanh((sq(x - xo) + sq(y - yo)- sq(rad))/Delta_int));
            u.x[] = U_inf*(1.0 - fs[]);
            u.y[] = 0;
            un = 0;
            foreach_dimension() un += u.x[] * normal.x;
            foreach_dimension() Us.x[] = u.x[] - un*normal.x;
        }
        boundary ({fs, Us, u.x, u.y});
        if (it>=10) printf("WARNING: does not converge... ");
    }while (adapt_wavelet({fs, u.x, u.y}, (double []){1e-3, 1e-3, 1e-3},
                          maxlevel = MAXLEVEL, minlevel = 3).nf != 0 && it <= 10);
    event ("end_timestep");
}

void exact_solution(vector ve, scalar pe){
    double theta, r, vr, vth;
    foreach() {
        r = sqrt(sq(x) + sq(y));
        theta = atan2(y, x + SEPS);
        vr  =   (r > rad) * U_inf * (1 - sq(rad/(r + SEPS))) * cos(theta);
        vth = - (r > rad) * U_inf * (1 + sq(rad/(r + SEPS))) * sin(theta);
//        vv = (r > rad)*U_inf * sqrt(1 + sq(sq(rad/(r + SEPS))) - 2.0 * sq(rad/(r+ SEPS)) * cos(2 * theta));
        ve.x[] = vr * cos(theta) - vth * sin(theta);
        ve.y[] = vr * sin(theta) + vth * cos(theta);
        pe[] = 0.5 * rho[] * (sq(U_inf) - sq(ve.x[]) - sq(ve.y[]));
    }
}

//Output
#include "../src_local/output_vtu_foreach.h"
event end_timestep (i += 1){
    char subname[80]; sprintf(subname, "br");
    scalar l[], pe[];
    vector ve[];
    exact_solution(ve, pe);
    vorticity (u, omega);
    foreach() l[] = level;
    output_vtu_MPI( (scalar *) {l, omega, fs, p, pe}, (vector *) {u, Us, dbp, ve}, subname, 0);
}

#define ADAPT_SCALARS {fs, omega}
#define ADAPT_EPS_SCALARS {1e-3, 1}
//Because of the spatio-temporal localization of our problem, grid adaptation is employed.
event adapt (i++){
    double eps_arr[] = ADAPT_EPS_SCALARS;
    MinMaxValues(ADAPT_SCALARS, eps_arr);
    adapt_wavelet ((scalar *) ADAPT_SCALARS, eps_arr, maxlevel = MAXLEVEL, minlevel = 4);
    boundary ({u.x, u.y});
}


event stop(t = 10);
