#define BRINKMAN_PENALIZATION 1
#define DEBUG_BRINKMAN_PENALIZATION 1
#define DEBUG_OUTPUT_VTU_MPI
#define FILTERED
//#define MODIFIED_CHORIN 1
#define JACOBI 1
#define EPS_MAXA 2
//#define SURFACE_TENSION 0
//#define FIXED
scalar omega[], my_kappa[];
scalar fs[];
face vector fs_face[];
//(const) face vector target_Uf = zerof;
#include "centered-weugene.h"
//#include "view.h"
#include "output_vtu_foreach.h"
//#include "../src_local/three-phase-weugene.h"//rho3, mu3
#include "two-phase.h"
#if SURFACE_TENSION == 1
	#include "tension.h"//f.sigma
#endif
int maxlevel = 9;
int minlevel = 4;
double U0 = 1, rhol = 1, sig = 0.0005, Lchar = 1, mul = 1, Lb = 0.3, Rb = 0.0625;
double Rrho = 1, Rmu = 1;
double RE, CA, Rrad;
double xs0 = -0.2, rad = 0.0625;
#ifdef FIXED
    coord vc = {0.0, 0.0, 0.0};
#else
    coord vc = {1.0, 0.0, 0.0};
#endif
double signvc = 1;
double deltaT = 0.0;
double thickness, Dmin, meps=0.01;
bool vtk_file_flag = false;

/**
The domain is the periodic unit square centered on the origin. */

int main(int argc, char * argv[])
{
    RE = U0*Lchar*rhol/mul;
    CA = U0*mul/sig;
    Rrad = Rb/rad;
    if (argc > 1) {
        RE = atof(argv[1]);
    }
    if (argc > 2) {
        CA = atof(argv[2]);
    }
    if (argc > 3) {
        Rrad = atof(argv[3]);
    }
    if (argc > 4) {
        Rrho = atof(argv[4]);
    }
    if (argc > 5) {
        Rmu = atof(argv[5]);
    }
    if (argc > 6) {
        maxlevel = atoi(argv[6]);
    }
    size (1.0);
    origin (-0.5*L0, -0.5*L0);
    eta_s = 1e-8;
    TOLERANCE = 1e-6;
    NITERMAX = 50;
    N = 1 << minlevel;
    periodic(top);
    periodic(right);

    Rb = Rrad*rad;
    rho1 = 1.; rho2 = rho1/Rrho; //rho3 = 10*max(rho1, rho2);
    mu1 = 1./RE; mu2 = mu1/Rmu; //mu3 = 10*max(mu1, mu2);
#if SURFACE_TENSION == 1
    f.sigma = 1./RE/CA;
#else
    fprintf(ferr,"NO SURFACE TENSION!\n");
#endif
    signvc = (vc.x > 0) ? 1 : (vc.x < 0)? -1 : 0;
    Dmin = L0*pow(2., -maxlevel);
    thickness = (2*rad + Dmin)*Dmin/atanh(1 - 2*meps);
#if SURFACE_TENSION == 1
    fprintf(ferr, "maxlevel=%d tol=%g NITERMAX=%d Dmin=%g thickness=%g\n"
                  "RE=%g CA=%g Rb/rad=%g rho1/rho2=%g mu1/mu2=%g\n"
                  "mu1=%g mu2=%g rho1=%g rho2=%g sigma=%g\n",
            maxlevel, TOLERANCE, NITERMAX, Dmin, thickness,
            RE, CA, Rrad, Rrho, Rmu,
            mu1, mu2, rho1, rho2, f.sigma);
#else
    fprintf(ferr, "maxlevel=%d tol=%g NITERMAX=%d Dmin=%g thickness=%g\n"
                  "RE=%g CA=%g Rb/rad=%g rho1/rho2=%g mu1/mu2=%g\n"
                  "mu1=%g mu2=%g rho1=%g rho2=%g sigma=0\n",
            maxlevel, TOLERANCE, NITERMAX, Dmin, thickness,
            RE, CA, Rrad, Rrho, Rmu,
            mu1, mu2, rho1, rho2);
#endif
    const vector U_sol[] = {vc.x, vc.y, vc.z};
    target_U = U_sol;
    const face vector U_solf[] = {vc.x, vc.y, vc.z};
    target_Uf = U_solf;

    run();
}

scalar divu[];
#define tmpposx (xs0 + vc.x*t)
#define outOfBox ((positionX < X0) || (positionX > X0 +L0))
double posx (double t){
    double positionX = tmpposx;
    while(outOfBox){
        positionX -= signvc*L0;
    }
    return positionX;
}
void soild_fs(scalar fs, face vector fs_face, double t){
    double tt = max(t-deltaT,0);
    double x00 = posx(tt);

    vertex scalar phi[];
    foreach_vertex() {
        phi[] = HUGE;
        for (int xi=-L0; xi <=L0; xi +=L0) {
            double x1 = x00 + xi;
            phi[] = intersection(phi[], (sq(x - x1) + sq(y - vc.y * tt) - sq(rad)));
        }
        phi[] = -phi[];
    }
    boundary ({phi});
    fractions (phi, fs, fs_face);
    fs.refine = fs.prolongation = fraction_refine;
    boundary({fs});
}
void bubbles (scalar f){
    vertex scalar phi[];
    face vector ff[];
    foreach_vertex() {
        phi[] = HUGE;
        for (int xi=-L0; xi <=L0; xi +=L0) {
            double x1 = xs0 + xi;
            phi[] = intersection(phi[], sq(x - x1) + sq(y) - sq(2.5*rad)  );
            phi[] = union(phi[], -sq(x - x1) - sq(y) + sq(1.05*rad) );
        }
    }
    boundary ({phi});
    fractions (phi, f, ff);
}
event init (t = 0) {
    if (!restore (file = "restart")) {
        int it = 0;
        do {
            soild_fs (fs, fs_face, 0);
            bubbles(f);
            foreach() {
#ifdef FIXED
                u.x[] = U0*(1.0 - fs[])*(fabs(deltaT) < SEPS);
#else
                u.x[] = U0*fs[]*(fabs(deltaT )< SEPS);
#endif
                u.y[] = 0;
            }
            boundary((scalar *){u});//fs inside solid_fs
            //call viscosity
        }while (adapt_wavelet({f, fs, u}, (double []){1e-5, 1e-5, 1e-2, 1e-2}, maxlevel=maxlevel, minlevel=minlevel).nf != 0 && ++it <= 10);
        refine((sq(x-xs0) + sq(y) <sq(1.2*rad)) && (sq(x-xs0) + sq(y) >sq(0.8*rad)) && level <10);
        DT = 1e-7;
        event("vtk_file");
    }
}

event set_dtmax (i++) {
    if (i<500) DT *= 1.05;
    if (DT>0.000047619) DT=0.00047619;
}

event vof(i++){
//    soild_fs(fs, fs_face, t);


//    soild_fs(fs, fs_face, t + 0.5*dt);
//    foreach() {
//        if (fabs(fs[] - 1) < SEPS) {
//            u.x[] = target_U.x[];
//            u.y[] = target_U.y[];
//        }
//        u.x[] = (1.0 - fs[])*u.x[] + fs[]*target_U.x[];
//    }

//    foreach_face() {
//            if (fabs(fs_face.x[] - 1) < SEPS) {
//                uf.x[] = target_Uf.x[];
//            }
//        uf.x[] = (1.0 - fs_face.x[])*uf.x[] + fs_face.x[]*target_Uf.x[];
//    }
//    boundary ((scalar*){uf});
//
//    double Linf_u = -10;
//
//    foreach (reduction(max:Linf_u)) {
//        divu[] = 0;
//        foreach_dimension() divu[] += (uf.x[1]-uf.x[])/Delta;
//        if (fabs(divu[]) > Linf_u) Linf_u = fabs(divu[]);
//    }
//    fprintf(ferr, "divu_max= %g", Linf_u);
//    if(vtk_file_flag){
//        event("vtk_file");
//        vtk_file_flag = false;
//    }

}
//event advection_term (i++){
////    soild_fs(fs, fs_face, t + 0.5*dt);
//}
//event viscous_term (i++){
//    soild_fs(fs, fs_face, t + dt);
//}
/*void correct_press(scalar p, int i){
    double press = 0;
    int ip = 0;
#if 1 // Left bottom Corner
    foreach(){
        if (ip == 0){
            press = p[];
            ip++;
            @if _MPI
                MPI_Bcast(&press, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            @endif
        }
    }
#else //average value
    press = normf(p).avg;
#endif
    foreach(){
        p[] -= press;
    }
}
*/

event end_timestep(i++){
//    correct_press(p, i);
    vorticity (u, omega);
}
double V0;
event end_timestep (i++) {
/*    double avggas = sq(L0) - normf(f).avg, avggas_out_solid = sq(L0) - normf_weugene(f, fs).avg,  u_mag;
    foreach() {
        divu[] = 0;
        foreach_dimension()
        divu[] += (uf.x[1]-uf.x[])/Delta;
    }
    fprintf (ferr, "i= %d t= %g dt= %g iter_p= %d iter_u= %d AvgGas= %17.14g V0= %17.14g divu= %15.12g "
    "Omega_min= %g Omega_max= %g u_min_mag= %g u_max_mag= %g "
    "u_min_x= %g u_max_x= %g u_min_y= %g u_max_y= %g \n",
    i, t, dt, avggas, Linf_u, Linf_omega_min, Linf_omega_max,
    u_min_mag, u_max_mag );
*/
}

/**
We produce animations of the vorticity and tracer fields... */

//event images (t += 0.1) {
//    static FILE * fp = popen ("ppm2gif > vort.gif", "w");
//    output_ppm (omega, fp, min=-10, max=10, linear=true);
//}

//Output
//event vtk_file (i>1000000){
//event vtk_file (i += 1){
event vtk_file (t += 0.01){
    vtk_file_flag = true;
    char subname[80]; sprintf(subname, "rk");
    scalar l[];
    foreach() {l[] = level;} boundary((scalar *){l});

#if SURFACE_TENSION == 1
	output_vtu_MPI( subname, (iter_fp) ? t + dt : 0, (scalar *) {fs, f, omega, rho, p, l, divu}, (vector *) {u, a});
#else
	output_vtu_MPI( subname, (iter_fp) ? t + dt : 0, (scalar *) {fs, f, omega, rho, p, l, divu}, (vector *){u});
#endif
}


#define ADAPT_SCALARS {f, fs, omega}
#define ADAPT_EPS_SCALARS {1e-5, 1e-5, 1e-2}
event adapt (i++){
    double eps_arr[] = ADAPT_EPS_SCALARS;
    MinMaxValues(ADAPT_SCALARS, eps_arr);
    adapt_wavelet ((scalar *) ADAPT_SCALARS, eps_arr, maxlevel = maxlevel, minlevel = minlevel);
    fs.refine = fs.prolongation = fraction_refine;
    boundary({fs});
}

//event snapshot (t += 0.1) {
//      char name[80];
//      sprintf (name, "snapshot-%g", t);
//      dump (name);
//}

event stop(t = 10);



//scalar divu[];
//double solid_function(double xc, double yc, double x, double y){
////    double xz=0.5*(1 - tanh((sq(x - xc) + sq(y - yc) - sq(rad))/thickness));
////    if (xz>0) fprintf(ferr, "xc= %g, yc= %g x= %g y= %g thickness= %g xz=%g\n", xc, yc, x, y, thickness, xz);
//    return 0.5*(1 - tanh((sq(x - xc) + sq(y - yc) - sq(rad))/thickness)); // 1 in cylindrical solid, 0 is outside
//}
//#define tmpposx (xs0 + vc.x*t)
//#define outOfBox ((positionX < X0) || (positionX > X0 +L0))
//double posx (double t){
//    double positionX = tmpposx;
//    while(outOfBox){
//        positionX -= signvc*L0;
//    }
//    return positionX;
//}
//void soild_fs(scalar fs, face vector fs_face, double t){
//    double tt = max(t-deltaT,0);
//    double x00 = posx(tt);
////    foreach(){
////        fs[] = 0;
////        for (int xi=-L0; xi <=L0; xi +=L0) {
////            double x1 = x00 + xi;
////            fs[] += solid_function(x1, vc.y * tt, x, y);
////        }
////    }
////    foreach_face(x) {
////        fs_face.x[] = 0;
////        for (int xi=-L0; xi <=L0; xi +=L0) {
////            double x1 = x00 + xi;
////            fs_face.x[] += solid_function(x1, vc.y * tt, x, y);
////        }
////    }
////    foreach_face(y) {
////        fs_face.y[] = 0;
////        for (int xi=-L0; xi <=L0; xi +=L0) {
////            double x1 = x00 + xi;
////            fs_face.y[] += solid_function(x1, vc.y * tt, x, y);
////        }
////    }
////    boundary((scalar *){fs_face});
//
//
//    vertex scalar phi[];
//    foreach_vertex() {
//        phi[] = HUGE;
//        for (int xi=-L0; xi <=L0; xi +=L0) {
//            double x1 = x00 + xi;
//            phi[] = intersection(phi[], (sq(x - x1) + sq(y - vc.y * tt) - sq(rad)));
//        }
//        phi[] = -phi[];
//    }
//    boundary ({phi});
//    fractions (phi, fs, fs_face);
//    foreach_face() {
//        fs_face.x[] = face_value(fs,0);
//    }
//    fs.refine = fs.prolongation = fraction_refine;
//    boundary({fs});
//}
//void bubbles (scalar f){
////    foreach(){
////        f[] = 0.5*(1 - tanh((sq(x - xs0) + sq(y) - sq(1.0*rad))/thickness));
////        f[] += 0.5*(tanh((sq(x - xs0) + sq(y) - sq(2.5*rad))/thickness) + 1);
////    }
////    boundary({f});
//    vertex scalar phi[];
//    face vector ff[];
//    foreach_vertex() {
//        phi[] = HUGE;
//        for (int xi=-L0; xi <=L0; xi +=L0) {
//            double x1 = xs0 + xi;
////            phi[] = intersection(phi[], sq(x - x1) + sq(y) > sq(2.5*rad)  ? 1 : -1);
////            phi[] = intersection(phi[], (sq(x - x1) + sq(y) > sq(2.5*rad) || sq(x - x1) + sq(y) < sq(1.05*rad) ) ? 1 : -1);
//            phi[] = intersection(phi[], sq(x - x1) + sq(y) - sq(2.5*rad)  );
//            phi[] = union(phi[], -sq(x - x1) - sq(y) + sq(1.0*rad) );
//        }
//    }
//    boundary ({phi});
//    fractions (phi, f, ff);
//}
