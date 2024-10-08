/**
# Two-phase flow around RV fiber

This is an improved version of the [famous Gerris
example](http://gerris.dalembert.upmc.fr/gerris/examples/examples/fiber.html),
illustrating the combination of complex solid boundaries, air-water
turbulent flows and reduced gravity approach.

We use the centered Navier--Stokes solver, two-phase flow and the
momentum-conserving option. Note that the momentum-conserving option
is crucial to obtain stable solutions for this air-water density ratio
configuration. */

//#include "grid/octree.h"
#include "grid/quadtree.h"
#include "navier-stokes/centered.h"// u, uf, p,  pf, g, mu=0, alphav=1, a=0(acceleration), rho=1,
// stokes=false (No omit div(u by u)
// incompressible NS.  generic time loop, a CFL-limited timestep,
// the Bell-Collela-Glaz (BCG) advection scheme ( 2nd-order, unsplit, upwind scheme)
// and the implicit viscosity solver
/**
We have two phases e.g. air and water. For large viscosity and density
ratios, the harmonic mean for the viscosity tends to work better than
the default arithmetic mean. We "overload" the default by defining the
*mu()* macro before including the code for two phases. */
#define mu(f)  (1./(clamp(f,0,1)*(1./mu1 - 1./mu2) + 1./mu2))
#include "two-phase.h" //advection of tracer f. defined rhov, alphav. if FILTERED is defined, then sf - filtered values of f.
#include "navier-stokes/conserving.h" //???should add?

/**
We also need surface tension, and in 3D only we will use the
$\lambda_2$ criterion of [Jeong and Hussain,
1995](/src/references.bib#jeong1995) to display the vortices using
Basilisk View. */

#include "tension.h"
#if dimension == 3
# include "lambda2.h"
#endif
#include "view.h"


/**
We also need to compute distance functions (to describe the ship
geometry), use reduced gravity and visualisation functions. */

//#include "distance.h"
#include "reduced.h"

/**
## Importing the geometry

This function computes the solid fraction given a pointer to an STL
file, a tolerance (maximum relative error on distance) and a
maximum level. */

//void fraction_from_stl (scalar f, FILE * fp, double eps, int maxlevel)
//{
//
//    /**
//    We read the STL file and compute the bounding box of the model. */
//
//    coord * p = input_stl (fp);
//    coord min, max;
//    bounding_box (p, &min, &max);
//    double maxl = -HUGE;
//    foreach_dimension()
//    if (max.x - min.x > maxl)
//        maxl = max.x - min.x;
//
//    /**
//    We initialize the distance field on the coarse initial mesh and
//    refine it adaptively until the threshold error (on distance) is
//    reached. */
//
//    scalar d[];
//    distance (d, p);
//    while (adapt_wavelet ({d}, (double[]){eps*maxl}, maxlevel, 5).nf);
//
//    /**
//    We also compute the volume fraction from the distance field. We
//    first construct a vertex field interpolated from the centered field
//    and then call the appropriate VOF functions. */
//
//    vertex scalar phi[];
//    foreach_vertex()
//    phi[] = (d[] + d[-1] + d[0,-1] + d[-1,-1] +
//             d[0,0,-1] + d[-1,0,-1] + d[0,-1,-1] + d[-1,-1,-1])/8.; //left bottom vertex
//    fractions (phi, f);
//}


int MAXLEVEL = 12;
int LEVEL = 12;
double FROUDE = 0.4;
double limMin, limMax;
static int iteration=0;
/**
The density ratio is 1000 and the dynamic viscosity ratio 100. */
#define RHOR 1000.
#define MUR 100.
#define feps 0.01
#define uemax 0.01
/**
We try to replicate the results of [Cano-Lozano et al,
2016](/src/references.bib#cano2016) (obtained with Gerris). Aside from
the ratios above, there are two independent parameters which can be
described by the Galilei number
$$
    Ga^2 = \frac{g D^3}{\nu^2}
$$
with $g$ the acceleration of gravity, $D$ the diameter of the bubble
and $\nu$ the kinematic viscosity of the outer fluid; and the
[Bond/Eötvös](https://en.wikipedia.org/wiki/E%C3%B6tv%C3%B6s_number)
number
$$
    Bo = \frac{\rho g D^2}{\sigma}
$$
with $\rho$ the density of the outer fluid and $\sigma$ the surface
tension coefficient.

 * We consider two bubbles studied by Cano-Lozano et al, 2016. */

#if BUBBLE19
// Bubble 19 of Cano-Lozano et al, P.R.Fluids, 2016
# define Ga 100.8
# define Bo 4.
# define MAXTIME 82
#else
// Bubble 26 of Cano-Lozano et al, P.R.Fluids, 2016
# define Ga 100.25
# define Bo 10.
# define MAXTIME 110
#endif

#define RandMinMax(limMin, limMax) (0.5*((limMin)+(limMax)) + 0.5*noise()*((limMax) - (limMin)))
#if dimension>2
# define mynorm(v) (sqrt(sq(v.x) + sq(v.y) + sq(v.z)))
#else
# define mynorm(v) (sqrt(sq(v.x) + sq(v.y)))
#endif
/**
 If the distance function is positive, then the value of $f=1$, otherwise it becomes 0.
 For example, at some point $f=1$ means that a point consist of pure liquid, and
 $f=0$ means that there is only gas in a point, no liquid.
 */

//void bubbles (scalar cs, face vector fs, double limMin, double limMax)
//{
//    const int ns = 30; // 80, 160, 165, 200
//    double xc[ns], yc[ns], R[ns], dist;
//#if dimension>2
//    double zc[ns];
//#endif
//    srand (0);
//    dist = sqrt(sq(limMax - limMin)+sq(L0)
//           #if dimension>2
//              +sq(L0)
//           #endif
//           )/(ns-1.0);
//    for (int i = 0; i < ns; i++){
//        R[i] = 0.03*L0*(fabs(noise())+0.1);
//        xc[i] = RandMinMax( limMin + R[i], limMax - R[i]);
//        yc[i] = RandMinMax(-7.0*R[i], 7.0*R[i]);
//        #if dimension>2
//            zc[i] = RandMinMax(-0.5*L0 + R[i], 0.5*L0 - R[i]);
//        #endif
//
//        for (int j = 1; j < i; j++) {
//            if (sq(xc[i]-xc[j]) + sq(yc[i]-yc[j])
//            #if dimension>2
//                +sq(zc[i]-zc[j])
//            #endif
//            <sq(dist)) i--;
//        }
//    }
//    /** Once we have defined the random centers and radii, we can compute
//    the levelset function $\phi$ representing the embedded boundary. */
//    vertex scalar phi[];
//    foreach_vertex() {
//        phi[] = HUGE;
//        /** Since the medium is periodic, we need to take into account all
//        the disk images using periodic symmetries. Periodic only along Oy, Oz */
//        for (double yp = -L0; yp <= L0; yp += L0)
//            for (double zp = -L0; zp <= L0; zp += L0)
//                for (int i = 0; i < ns; i++)
//                    phi[] = intersection (phi[], (sq(x - xc[i]) + sq(y + yp - yc[i])
//#if dimension>2
//                            + sq(z + zp - zc[i])
//#endif
//                            - sq(R[i])));
//    }
//    fractions (phi, cs, fs);
//}


double cubeF(double x, double y, double z, coord center, double size) {
//    Our cube is defined as the intersection of 6 orthogonal planes. We define first 2 planes,  and .
//
//    Then we define P1 has:
    double P1_Plus = x - size/2. + center.x;
    double P1_Minus = x + size/2. + center.x;
    double P1 = max (P1_Plus, - P1_Minus);
//    We apply the same process to obtain P2 and P3
    double P2_Plus = y - size/2. + center.y;
    double P2_Minus = y + size/2. + center.y;
    double P2 = max (P2_Plus, -P2_Minus);
#if dimension>2
    double P3_Plus = z - size/2. + center.z;
    double P3_Minus = z + size/2. + center.z;
    double P3 = max (P3_Plus, -P3_Minus);
    double c = max ( P1, max (P2, P3) );
#else
    double c = max ( P1, P2 );
#endif
    return -c;
}
double sphere(double x, double y, double z, coord center, double radius) {
    return ( sq(x - center.x) + sq (y - center.y) + sq (z - center.z)
             - sq (radius));
}

double bubbles (double x, double y, double z)
{
    const int ns = 10;
    coord mypoint = {x,y,z};
    coord pnt_dist;
    coord centers[ns];
    double R[ns], dist=1.1*sqrt(sq(limMax - limMin)+sq(L0)
#if dimension>2
            +sq(L0)
#endif
    )/(ns+2.0);;

    srand (0);
    int iter = 0, i = 0;
    while(i<ns){
        R[i] = 0.01*L0*(fabs(noise())+1.0);
        centers[i].x = RandMinMax( limMin + R[i], limMax - R[i]);
        centers[i].y = RandMinMax(-L0/2.0, L0/2.0);
#if dimension>2
        centers[i].z = RandMinMax(-0.5*L0 + R[i], 0.5*L0 - R[i]);
#endif
        for (int j = 1; j < i; j++) {
            foreach_dimension()
            pnt_dist.x = centers[i].x - centers[j].x;
            if (mynorm(pnt_dist) < dist) {
                i--;
                break;
            };
        }
        i++;
        iter++;
        if (iter>100*ns) exit(137);
    }

    for (int i = 0; i < ns; i++){
        R[i] = 0.02*L0*(fabs(noise())+1.0);
        centers[i].x = RandMinMax( limMin + R[i], limMax - R[i]);
        centers[i].y = RandMinMax(-0.5*L0, 0.5*L0);
//        centers[i].y = RandMinMax(-7.0*R[i], 7.0*R[i]);
#if dimension>2
        centers[i].z = RandMinMax(-0.5*L0 + R[i], 0.5*L0 - R[i]);
#endif
        for (int j = 0; j < i; j++) {
            foreach_dimension()
            pnt_dist.x = centers[i].x - centers[j].x;
            if (mynorm(pnt_dist) < dist) {
                i--;
                break;
            };
        }
    }



    double phi = HUGE;
    for (int i = 0; i < ns; i++) {
        //printf("i=%d x=%g y=%g R=%g\n", i, centers[i].x, centers[i].y, R[i] );
        foreach_dimension()
            pnt_dist.x = mypoint.x - centers[i].x;
        phi = min(phi, (mynorm(pnt_dist) - R[i]));
    }
    return min(phi, -L0/8.0 - x);// with front
//    return phi; // no front
}


//center.x = 0;
//center.y = 0;
//#if dimension > 2
//    center.z = 0;
//#endif
coord center={0,0,0};
const int Ncyl=7;
double size_box;
double R;//0.5
double dist;

double cylindersOy(double x, double y, double z, double Ox) {
    int icyl;
//    We define the 2*Ncyl cylinders along the ,  and  axis.
    double cylinderX[Ncyl], tmp;
#if dimension>2
    double  cylinderZ[Ncyl];
#endif
    for (icyl=0; icyl<Ncyl; icyl++){
//        tmp = dist*(icyl-(Ncyl-1.0)/2.0);
        tmp = dist*icyl + center.y - 0.5*size_box + R;
//        cylinderX[icyl] = sq(R)- sq(y+2.0*R) - sq(x-tmp);
        cylinderX[icyl] = sq(R)- sq(x + 2.0*R - Ox) - sq(y - tmp);
#if dimension>2
        cylinderZ[icyl] = sq(R) - sq(y) - sq(z-tmp) ;
#endif
    }
    //    We use an intermediate object for the union (to decompose the process)
    double geom = cylinderX[0];
    for (icyl=1; icyl<Ncyl; icyl++){
        geom = max(cylinderX[icyl], geom);
    }
#if dimension>2
    for (icyl=0; icyl<Ncyl; icyl++){
        geom = max(cylinderZ[icyl], geom);
    }
#endif
    return geom;
}

double geometry(double x, double y, double z) {
    const int Nlayer=4;
    int icyl;
    double dist = 0.5*L0/Nlayer;
    double geom = cylindersOy(x, y, z, 0.0);

    for (icyl=1; icyl<Nlayer; icyl++){

        geom = max(cylindersOy(x, y, z, icyl*dist), geom);
    }
    double c = cubeF(x, y, z, center, size_box);
    return min(geom,c);
}
/**
## Main function

We can change both the maximum level of refinement and the [Froude
number](https://en.wikipedia.org/wiki/Froude_number) at runtime.

[RV fiber](https://en.wikipedia.org/wiki/RV_fiber) is 70 metres
long. If we assume that it moves at 20 knots (twice its actual cruise
speed), this gives a Froude number of approx 0.4. */


/**
We need additional (fraction) fields for the ship geometry and for the
(inflow) boundary condition. */

scalar fiber[], f0[];
FILE *fplog;

int main (int argc, char * argv[])
{
//    maxruntime (&argc, argv);
//    if (argc > 1) {
//        LEVEL = atoi(argv[1]); //convert from string to int
//    }
//    if (argc > 2) {
//        FROUDE = atof(argv[2]);
//    }
    periodic (top);
#if dimension>2
    periodic (back);
#endif
    init_grid (64);
    rho1 = 1.;// water
    rho2 = 1./RHOR; // air
    mu1 = 1./Ga;
    mu2 = 1./(MUR*Ga);
    f.sigma = 1./Bo;


    /**
    The length of the ship is unity and the domain is five times
    larger. We change the origin so that the ship is not too close to
    the inflow. */
    stokes = true;
    size (4);
    origin (-L0/2.,-L0/2.,-L0/2.);
    limMin =-0.5*L0;
    limMax =-0.25*L0;
    /**
  We reduce the tolerance on the divergence of the flow. This is
  important to minimise mass conservation errors for these simulations
  which are very long. */

    TOLERANCE = 1e-4;
    /**
    We need to tell the code that both `fiber` and `f0` are volume
    fraction fields. */

    for (scalar s in {fiber,f0}) {
        s.refine = s.prolongation = fraction_refine;
    }

    /**
    Since the ship length is one and the velocity one, the acceleration
    of gravity is...*/
    fplog = fopen("fplog.txt", "w");
    run();
}

/**
## Boundary conditions

The inflow condition fixes the velocity (unity) and the water level
(using `f0`). */



u.n[left] = dirichlet(1.);
p[left]   = neumann(0.);
pf[left]  = neumann(0.);
f[left]   = dirichlet(1.0); //liquid
//fiber[left] = dirichlet(0);

u.n[right] = neumann(0.);
p[right]   = dirichlet(0.);
pf[right]  = dirichlet(0.);
f[right]   = neumann(0.);//liquid
//fiber[right] = dirichlet(0);
/**
Outflow uses standard Neumann/Dirichlet conditions.  */

//u.n[top]  = neumann(0.);
//p[top]    = dirichlet(0.);
//pf[top]   = dirichlet(0.);

/**
Boundary conditions for the solid and fraction tracers. */

//fiber[back] = 0;
//f[back] = 1;

/**
Not sure whether this is really useful. */

//uf.n[left] = 0.;
//uf.n[right] = 0.;

/**
## Initial conditions

We can optionally restart, otherwise we open the STL file and
initialize the corresponding fraction. We also initialize the `f0`
field used for the inflow condition and set the initial water level
and velocity field. */

event init (t = 0) {
    size_box = 0.8*L0;
    R = size_box*0.2/(max(Ncyl-1,1));//0.5
    dist = (size_box-2.0*R)/(max(Ncyl-1,1));

    if (!restore (file = "restart")) {
//        from stl file
//        FILE * fp = fopen ("fiber.stl", "r");
//        fraction_from_stl (fiber, fp, 5e-4, MAXLEVEL);
//        fclose (fp);
        int iteration = 0;
        do {
            iteration++;
            fraction(fiber, geometry (x, y, z));
            fraction(f0, bubbles (x, y, z));
//            bubbles (f0, fs, -0.5*L0, -0.25*L0);
        }while (adapt_wavelet({fiber, f0}, (double []){0.01, 0.01},
                              maxlevel = MAXLEVEL, 2).nf != 0 && iteration <= MAXLEVEL);

        foreach() {
            f[] = f0[] + fiber[];// perfect fixed wetting drops instead of solid cylinders
            u.x[] = 1.;
        }
        boundary ({f,u.x, u.y}); //?
#if dimension == 2
        view (tx = 0, width = 2048, height = 2048, samples = 4, bg = {0.3,0.4,0.6});
  clear();
  draw_vof ("f", edges = true);
  draw_vof("fiber", edges = true);
  box ();
  save ("vof2D.png");
#else // 3D
//        view (fov = 20, camera = "iso",//quat = {-0.4,0.4,0.1,0.8},
//                tx = 0, ty = 0,
//                bg = {0.3,0.4,0.6},
//                width = 2048, height = 2048, samples = 4);
//        clear();
//        draw_vof("f", edges = true);
//        draw_vof("fiber", edges = true);
//        box();
//        save ("vof.png");

//        clear();
//        view (fov = 40, camera = "top",//quat = {-0.4,0.4,0.1,0.8},
//              tx = 0, ty = 0,
//              bg = {0.3,0.4,0.6},
//              width = 2048, height = 2048, samples = 4);
//
//        draw_vof("f", edges = true);
//        draw_vof("fiber", edges = true);
//        box();
//        save ("vof_top.png");

        clear();
        view (fov = 40, camera = "front",//quat = {-0.4,0.4,0.1,0.8},
              tx = 0, ty = 0,
              bg = {0.3,0.4,0.6},
              width = 2048, height = 2048, samples = 4);

        draw_vof("f", edges = true);
        draw_vof("fiber", edges = true);
        box();
        save ("vof_front.png");
#endif // 3D
    }
}

/**
## Boundary conditions on the ship

We use a simple (but crude) imposition of $u=0$ in the solid. */

event velocity (i++) {
    foreach()
    {
        foreach_dimension()
            u.x[] = (1. - fiber[]) * u.x[];//velocity = 0 in the solid
    }
    boundary ((scalar *){u});

}

event logfile (i++)
if (pid()==0) fprintf (fplog, "%d %g %d %d\n", i, t, mgp.i, mgu.i);

/**
## Animations

We generate animations of the ship surface (as represented by the
solid fraction) and of the air-water interface, colored with the
height.

Several classical features of ship wakes are recognisable: breaking
bow wave, breaking stern divergent wave, turbulent boundary layer,
Kelvin waves etc...

![Evolution of the air-water interface](fiber/movie.mp4)(width="800" height="600")

We also use the $\lambda_2$ criterion to display the turbulent
vortical structures generated by the airflow. The air recirculation at
the top of the steep primary Kelvin waves is particularly noticeable.

![Turbulent vortical structures](fiber/l2.mp4)(width="800" height="600")

The computations above were done on the Irene supercomputer using 12
levels of refinement. */

//event movie (t += 0.01; t <= 10)
//{
//#if dimension == 2
//view (tx = 0, width = 2048, height = 2048, samples = 4, bg = {0.3,0.4,0.6});
//    clear();
//    draw_vof ("f", edges = true);
//    draw_vof("fiber", edges = true);
//    box ();
//    save ("vof2D.mp4");
//#else // 3D
//clear();
//view (fov = 40,
//        quat = {0.5,0.1,0.2,0.8},
//        tx = 0, ty = 0,
//        width = 1600, height = 1600);
//draw_vof ("fiber");
//draw_vof ("f", linear = true);
//save ("movie.mp4");
//#endif

//    static FILE * fpall = fopen ("all_fields", "w");
//    output_ppm (f, fpall, min = 0, max = 1);

//output_vtk_unstructured({f, fiber, u.x, u.y, rhov}, 1>>MAXLEVEL, fplist, false );
//draw_vof ("fiber", fc = {0.5,0.5,0.5});//fiber is grey
//draw_vof ("f", color = "Z", min = -0.1, max = 0.1, linear = true);
//scalar l2[];
//lambda2 (u, l2);
//isosurface ("l2", -100);
//save ("l2.mp4");
//}

#if _MPI != 1
#include "vtk.h"
//#include "output_vtk.h"
event vtk_file (i += 100)
{
    char name[80];
    sprintf(name, "list.vtk.%d", iteration++);
    printf("name %s", name);
    FILE * fplist = fopen (name, "w");
    output_vtk ({f, fiber, u.x, u.y, rhov}, 64, fplist, false );
    //output_vtk_unstructured ({f, fiber, u.x, u.y, rhov, p, pf}, 512, fplist, false );
}
#endif


event grid_file (i += 1000){
    scalar l[];
    foreach()
    l[] = level;
    static FILE * fp = fopen ("grid.ppm", "w");
    output_ppm (l, fp, min = 0, max = LEVEL, n=1024);
}


#if DUMP
event snapshot (t += 0.01)
//event snapshot (i += 1000)
//event snapshot (i++)
{
  char name[80];
  sprintf (name, "dump-%g", t);
  scalar l2[];

//  lambda2 (u, l2);
  dump (file = name);
}
#endif




//Output
#include "output_fields/output_vtu_foreach.h"
event vtk_file (t += 0.01; t<10)
{
    int nf = iteration;
    scalar l[];
    foreach()
        l[] = level;

    char name[80], subname[80];
    FILE *fp;
    sprintf(name, "hs_%4.4d_n%3.3d.vtu", nf, pid());
    fp = fopen(name, "w");

    output_vtu_bin_foreach((scalar *) {l, f, fiber, p}, (vector *) {u}, 64, fp, false);
    fclose(fp);
    @if _MPI
    if (pid() == 0) {
    sprintf(name, "hs_%4.4d.pvtu", nf);
    sprintf(subname, "hs_%4.4d", nf);
    fp = fopen(name, "w");
    output_pvtu_bin((scalar *) {l, f, fiber, p}, (vector *) {u}, 64, fp, subname);
    fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    @endif
            fprintf (ferr, "iteration: %d\n", iteration); fflush (ferr);
    iteration++;
}

/**
## Mesh adaptation

This computation is only feasible thanks to mesh adaptation, based
both on volume fraction and velocity accuracy. */

event adapt (i++) {
    adapt_wavelet ({f,fiber,u},
                   (double[]){feps,feps,uemax, uemax, uemax}, maxlevel=MAXLEVEL, minlevel=5);
}
