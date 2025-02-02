#ifndef BASILISK_HEADER_12
#define BASILISK_HEADER_12
#line 1 "./../src_local/../src_local/three-phase-rheology.h"
/**
# Three-phase interfacial flows

This file helps setup simulations for flows of two fluids separated by
an interface (i.e. immiscible fluids) with solid obstacles. It is typically used in
combination with a [Navier--Stokes solver](navier-stokes/centered.h). 

The interface between the fluids is tracked with a Volume-Of-Fluid
method. The volume fraction $f$ and $fs$ leads to next interpretation of averaged value of A =(\rho, \mu, \kappa)
\begin{table}[]
\begin{tabular}{lll}
f & fs & A    \\
1 & 1  & A\_3 \\
0 & 1  & A\_3 \\
1 & 0  & A\_1 \\
0 & 0  & A\_2
\end{tabular}
\end{table}
 The above definition of variables leads to a specific definition of the averaged value A.
The densities and dynamic viscosities for fluid 1 and 2 are *rho1*,
*mu1*, *rho2*, *mu2*, respectively, in a solid rho3 and mu3.*/

#include "vof.h"
double VOF_cutoff = 0.01;
scalar f[], fs[], * interfaces = {f}, * interfaces_all = {f,fs};
double rho1 = 1., mu1 = 0., rho2 = 1., mu2 = 0., rho3 = 1., mu3 = 0.;
double kappa1 = 0, kappa2 = 0, kappa3 = 0;//W/(m*K)

/**
Auxilliary fields are necessary to define the (variable) specific
volume $\alpha=1/\rho$ as well as the cell-centered density. */


face vector alphav[];
face vector kappav[];
scalar rhov[];

event defaults (i = 0) {
  alpha = alphav;
  rho = rhov;

  /**
  If the viscosity and conductivity are non-zero, we need to allocate the face-centered
  viscosity and conductivity fields. */
  
  if (mu1 || mu2) //?
    mu = new face vector;

  if (kappa1 || kappa2) //?
      kappa = new face vector;
}

/**
The density, viscosity and conductivity are defined using arithmetic averages by
default. The user can overload these definitions to use other types of
averages (i.e. harmonic).
Usually, it is assumed that mu1 is variable, mu2 and mu3 are not. For simplisity mu3=mu2
 */
#ifndef rho
#define rho(f, fs) (clamp(f,0.,1.)*(rho1 - rho2) + rho2 + clamp(fs,0.,1.)*(rho3 - rho2))
//#define rho(f, fs) (clamp(f-fs,0.,1.)*(rho1 - rho2) + rho2 + clamp(fs,0.,1.)*(rho3 - rho2))
#endif

#ifndef kappav
#define kappav(f, fs)  (clamp(f,0.,1.)*(kappa1 - kappa2) + kappa2 + clamp(fs,0.,1.)*(kappa3 - kappa2))
//#define kappav(f, fs)  (clamp(f-fs,0.,1.)*(kappa1 - kappa2) + kappa2 + clamp(fs,0.,1.)*(kappa3 - kappa2)) //CORRECT IT !!!
//#define kappav(f, fs)  (1./(  clamp(f-fs,0.,1.)*(1./kappa1 - 1./kappa2) + 1./kappa2 + fs*(1./kappa3 - 1./kappa2)  )    )
#endif
/**
# Variable rheology models
 $$\mu = \mu_0 \exp(\frac{E_\eta}{RT})(\frac{\alpha_{gel}}{\alpha_{gel}-\alpha})^f(\alpha, T)$$
 $$f(\alpha, T) = A + B \alpha$$

 $$\mu = \mu_0 \exp(\frac{E_\eta}{RT}+\chi \alpha^2)$$
 **/
scalar alpha_doc[];
scalar T[];
double Eeta_by_Rg = 0.1; //Kelvin
double chi = 1;
double alpha_gel = 0.8;
#ifndef fpol
#define fpol(alpha_doc, T) A*alpha_doc + B
#endif

#ifndef muf1
//#define mupol(alpha_doc, T) (mu1*exp(Eeta_by_Rg/(T))*pow(alpha_gel/(alpha_gel-alpha_doc), fpol(alpha_doc, T)))
#define muf1(alpha_doc, T) (mu1*exp(Eeta_by_Rg/T+chi*alpha_doc*alpha_doc))
#endif

#ifndef mu
#define mu(f, fs, alpha_doc, T)  (clamp(f,0.,1.)*(muf1(alpha_doc, T) - mu2) + mu2 + clamp(fs,0.,1.)*(mu3 - mu2))
//#define mu(f, fs, alpha_doc, T)  (clamp(f-fs,0.,1.)*(mu1 - mu2) + mu2 + clamp(fs,0.,1.)*(mu3 - mu2)) //CORRECT IT !!!
//#define mu(f, fs, alpha_doc, T)  (clamp(f-fs,0.,1.)*(muf1(alpha_doc, T) - mu2) + mu2 + fs*(mu3 - mu2))
//#define mu(f, fs, alpha_doc, T)  (1./(  clamp(f-fs,0.,1.)*(1./muf1(alpha_doc, T) - 1./mu2) + 1./mu2 + fs*(1./mu3 - 1./mu2)  )    )
#endif


/**
We have the option of using some "smearing" of the density/viscosity
jump. */

#ifdef FILTERED
scalar sf1[], sf2[];
#else
#define sf1 f
#define sf2 fs
#endif
scalar *smearInterfaces = {sf1,sf2};

event properties (i++) {
  /**
  When using smearing of the density jump, we initialise *sf* with the
  vertex-average of *f*. */

#ifdef FILTERED
  int counter1 = 0;
  for (scalar sf in smearInterfaces){
    counter1++;
    int counter2 = 0;
    for (scalar f in interfaces_all){
      counter2++;
      if (counter1 == counter2){
        // fprintf(ferr, "%s %s\n", sf.name, f.name);
      #if dimension <= 2
          foreach(){
            sf[] = (4.*f[] +
        	    2.*(f[0,1] + f[0,-1] + f[1,0] + f[-1,0]) +
        	    f[-1,-1] + f[1,-1] + f[1,1] + f[-1,1])/16.;
          }
      #else // dimension == 3
          foreach(){
            sf[] = (8.*f[] +
        	    4.*(f[-1] + f[1] + f[0,1] + f[0,-1] + f[0,0,1] + f[0,0,-1]) +
        	    2.*(f[-1,1] + f[-1,0,1] + f[-1,0,-1] + f[-1,-1] +
        		f[0,1,1] + f[0,1,-1] + f[0,-1,1] + f[0,-1,-1] +
        		f[1,1] + f[1,0,1] + f[1,-1] + f[1,0,-1]) +
        	    f[1,-1,1] + f[-1,1,1] + f[-1,1,-1] + f[1,1,1] +
        	    f[1,1,-1] + f[-1,-1,-1] + f[1,-1,-1] + f[-1,-1,1])/64.;
          }
      #endif
      }
    }
  }
#endif

#if TREE
    for (scalar sf in smearInterfaces){
        sf.prolongation = refine_bilinear;
        boundary ({sf});
    }
#endif
  foreach_face() {
    double ff1 = (sf1[] + sf1[-1])/2.;
    double ff2 = (sf2[] + sf2[-1])/2.; //solid
    alphav.x[] = fm.x[]/rho(ff1, ff2);
    if (mu1 || mu2) {
      face vector muv = mu;
      muv.x[] = fm.x[]*mu(ff1, ff2, alpha_doc[], T[]);
    }
    if (kappa1 || kappa2) {
        face vector kappav = kappa;
        kappav.x[] = fm.x[]*kappav(ff1, ff2);
    }
  }
  foreach()
    rhov[] = cm[]*rho(sf1[], sf2[]); //? alphav.x and rhov are not consistent - All do so

#if TREE
  for (scalar sf in smearInterfaces){
     sf.prolongation = fraction_refine;
     boundary ({sf});
  }
#endif
}

#endif
