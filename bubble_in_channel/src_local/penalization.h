#ifndef BASILISK_HEADER_32
#define BASILISK_HEADER_32
#line 1 "./../src_local/./../src_local/penalization.h"
const vector zerocf[] = {0.,0.,0.};
#ifdef BRINKMAN_PENALIZATION
    #define frhs 1
    #define fbp (fs[])
    extern scalar fs;
    double eta_s = 1e-15, nu_s = 1, lambda_slip = 0;
    (const) scalar a_br = unity, b_br = unity; // useful for Robin BC
    (const) vector U_solid = zerocf;
    #if BRINKMAN_PENALIZATION == 1 //Dirichlet BC
        (const) vector target_U = zerocf, n_sol = zerocf;
        #define PLUS_BRINKMAN_RHS         + fbp*dt*( - (u.x[] - target_U.x[])/eta_s )
        #define PLUS_NUMERATOR_BRINKMAN   + 0
        #define PLUS_DENOMINATOR_BRINKMAN + fbp*dt*(sq(Delta)/eta_s)
    #elif BRINKMAN_PENALIZATION == 2 //Neumann BC
        vector target_U[], n_sol[];
        #define CALC_GRAD
        #define PLUS_BRINKMAN_RHS         + fbp*dt*(nu_s*(u.x[1] - 2*u.x[] +u.x[-1])/sq(Delta) - (scalar_a_by_b(n_sol, grad_u) - target_U.x[])/eta_s)
        #define PLUS_NUMERATOR_BRINKMAN   + fbp*dt*(nu_s*(u.x[1] + u.x[-1]) - (0.5*Delta*(m_scalar_a_by_b(n_sol, u)) - sq(Delta)*target_U.x[])/eta_s)
        #define PLUS_DENOMINATOR_BRINKMAN + fbp*dt*(2*nu_s)
    #elif BRINKMAN_PENALIZATION == 3 //Robin BC
        vector target_U[], n_sol[];
        #define CALC_GRAD
        #define PLUS_BRINKMAN_RHS         + fbp*dt*(nu_s*(u.x[1] - 2*u.x[] +u.x[-1])/sq(Delta) - (a_br[]*u.x[] + b_br[]*scalar_a_by_b(n_sol, grad_u) - target_U.x[])/eta_s)
        #define PLUS_NUMERATOR_BRINKMAN   + fbp*dt*(nu_s*(u.x[1] + u.x[-1]) - (0.5*Delta*b_br[]*(m_scalar_a_by_b(n_sol, u)) - sq(Delta)*target_U.x[])/eta_s)
        #define PLUS_DENOMINATOR_BRINKMAN + fbp*dt*(a_br[]*sq(Delta)/eta_s + 2*nu_s)
    #elif BRINKMAN_PENALIZATION == 4 //No penetration & slip BC
        vector target_U[], n_sol[]; //here target_U will be recalcilated each time step
        #define CALC_GRAD_U_TAU
        #define PLUS_BRINKMAN_RHS         + fbp*dt*(- (u.x[] - target_U.x[])/eta_s)
        #define PLUS_NUMERATOR_BRINKMAN   + fbp*dt*(sq(Delta)*target_U.x[]/eta_s)
        #define PLUS_DENOMINATOR_BRINKMAN + fbp*dt*(sq(Delta)/eta_s)
//    #elif BRINKMAN_PENALIZATION == 5//Ideal slip, no friction
//        #define CALC_GRAD
//        #define PLUS_BRINKMAN_RHS         + fbp*dt*(nu_s*(u.x[1] - 2*u.x[] +u.x[-1])/sq(Delta) - n_sol.x[]*scalar_a_by_b(n_sol, umUs)/eta_s)
//        #define PLUS_NUMERATOR_BRINKMAN   + fbp*dt*(nu_s*(u.x[1] + u.x[-1]) - sq(Delta)*n_sol.x[]*(scalar_a_by_b(n_sol, umUs) - n_sol.x[]*u.x[])/eta_s)
//        #define PLUS_DENOMINATOR_BRINKMAN + fbp*dt*(sq(n_sol.x[]*Delta)/eta_s + 2*nu_s)
    #else
        #define BRINKMAN_PENALIZATION_ERROR_BC 1
    #endif
    #undef SEPS
    #define SEPS 1e-15
//    #ifdef CALC_GRAD
//
//    #endif
    #ifdef DEBUG_BRINKMAN_PENALIZATION
        vector dbp[], total_rhs[], utau[], grad_utau_n[];
        #define gradun grad_utau_n.x[]
    #else
        double gradun;
    #endif
#else
    #define frhs 1
    #define fbp 0
    double gradun;
    #define PLUS_BRINKMAN_RHS 0
    #define PLUS_NUMERATOR_BRINKMAN 0
    #define PLUS_DENOMINATOR_BRINKMAN 0
#endif


struct Brinkman {
    vector u;
    face vector uf;
    scalar rho;
    double dt;
};


void calc_target_U(const vector u, vector target_U, const vector normal){
    if (!is_constant(U_solid.x)) foreach() foreach_dimension() target_U.x[] = U_solid.x[];
    if (fabs(lambda_slip) > 0.) {
        double ubyn;
        #ifndef DEBUG_BRINKMAN_PENALIZATION
        vector utau[]; // otherwise utau will be defined globally
        #endif
        if (!is_constant(U_solid.x)) foreach() foreach_dimension() u.x[] -= U_solid.x[];
        //    if (!is_constant(U_solid.x)) fprintf(ferr, "U_solid.x");
        foreach() {
            ubyn = 0;
            foreach_dimension() ubyn += (u.x[])*normal.x[];
            foreach_dimension() utau.x[] = u.x[] - ubyn*normal.x[];
        }
        if (!is_constant(U_solid.x)) foreach() foreach_dimension() u.x[] += U_solid.x[];
        foreach() {
            foreach_dimension() {
                if (0 < fs[] ) { //See here!
                    gradun = ((1 - fs[-1])*utau.x[-1] - (1 - fs[-2])*utau.x[-2] + (1 - fs[2])*utau.x[2] - (1 - fs[1])*utau.x[1])*n_sol.x[] / Delta
                            #if dimension > 1
                            + ((1 - fs[0,-1])*utau.x[0,-1] - (1 - fs[0,-2])*utau.x[0,-2] + (1 - fs[0,2])*utau.x[0,2] - (1 - fs[0,1])*utau.x[0,1])*n_sol.y[]/Delta
                            #endif
                            #if dimension > 2
                            + ((1 - fs[0, 0, -1])*utau.x[0, 0, -1] - (1 - fs[0, 0, -2])*utau.x[0, 0, -2] +  (1 - fs[0, 0, 2])*utau.x[0, 0, 2] - (1 - fs[0, 0, 1])*utau.x[0, 0, 1])*n_sol.z[] / Delta
                            #endif
                            ;
                } else {
                    gradun = 0;
                }
                target_U.x[] += lambda_slip*gradun;
                //            if (target_U.x[]) fprintf(ferr, "Ut = %g, U_s = %g, l=%g gr=%g\n", target_U.x[], U_solid.x[], lambda_slip, gradun);
            }
        }
    }
}

void brinkman_correction_u (vector u, double dt){
    if (!is_constant(target_U.x)) calc_target_U(u, target_U, n_sol);
    foreach() {
        foreach_dimension(){
            u.x[] = (u.x[] + (fbp*dt/eta_s)*target_U.x[])/(1. + fbp*dt/eta_s);
#ifdef DEBUG_BRINKMAN_PENALIZATION
            dbp.x[] = (PLUS_BRINKMAN_RHS)/dt;
#endif
        }
    }
    boundary ((scalar *){u});
}

void brinkman_correction_uf (face vector uf, double dt){
    double fs_face, target_U_face;
    foreach_face(){
        fs_face = face_value(fs, 0);
        target_U_face = (((fs[-1] > 0)*target_U.x[-1] + (fs[] > 0)*target_U.x[])/((fs[-1] > 0) + (fs[] > 0) + 1e-20));//face_value(target_U.x, 0);
        uf.x[] = (uf.x[] + (fs_face*dt/eta_s)*target_U_face)/(1 +fs_face*dt/eta_s);
    }
    boundary ((scalar *){uf});
}

void brinkman_correction (struct Brinkman p){
    vector u = p.u; face vector uf = p.uf; double dt = p.dt;
    brinkman_correction_u (u, dt);
//    brinkman_correction_uf (uf, dt);
}



//void calc_target_Uf(face vector uf, vector U_solid, face vector target_U, vector normal){
//    double ubyn;
//#ifndef DEBUG_BRINKMAN_PENALIZATION
////    face vector utau[]; // otherwise utau will be defined globally
//    face vector uftau[], fs_face[], n_sol_face[], U_solid_face[];
//#endif
//    foreach_face() {
//        fs_face.x[] = face_value(fs, 0);
//        n_sol_face.x[] = face_value(normal.x, 0);
//        uf.x[] -= face_value(U_solid.x,0);
//    }
//    boundary ((scalar *){fs_face, n_sol_face});
//    foreach_face() {
//        ubyn = 0; foreach_dimension() ubyn += uf.x[]*n_sol_face.x[];
//        foreach_dimension() uftau.x[] = uf.x[] - ubyn*n_sol_face.x[];
//    }
//
//    foreach_face() {
//        if (0 < fs[] && fs[] < 0.5){
//            gradun = ((1 - fs[-1])*utau.x[-1] - (1 - fs[-2])*utau.x[-2] + (1 - fs[2])*utau.x[2] - (1 - fs[1])*utau.x[1])*n_sol.x[]/Delta
//#if dimension > 1
//                + ((1 - fs[0,-1])*utau.x[0,-1] - (1 - fs[0,-2])*utau.x[0,-2] + (1 - fs[0,2])*utau.x[0,2] - (1 - fs[0,1])*utau.x[0,1])*n_sol.y[]/Delta
//#endif
//#if dimension > 2
//                + ((1 - fs[0, 0, -1])*utau.x[0, 0, -1] - (1 - fs[0, 0, -2])*utau.x[0, 0, -2] +  (1 - fs[0, 0, 2])*utau.x[0, 0, 2] - (1 - fs[0, 0, 1])*utau.x[0, 0, 1])*n_sol.z[] / Delta
//#endif
//                    ;
//        }else{
//            gradun = 0;
//        }
//        target_U.x[] = U_solid.x[] + lambda_slip*gradun;
//    }
//    foreach_face() {
//        uf.x[] += face_value(U_solid.x,0);
//    }
//}

#endif
