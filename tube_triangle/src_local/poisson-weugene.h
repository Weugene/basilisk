#ifndef BASILISK_HEADER_35
#define BASILISK_HEADER_35
#line 1 "./../src_local/poisson-weugene.h"
    /**
# Multigrid Poisson--Helmholtz solvers

We want to solve Poisson--Helmholtz equations of the general form
$$
L(a) = \nabla\cdot (\alpha\nabla a) + \lambda a = b
$$
This can be done efficiently using a multigrid solver. 

An important aspect of Poisson--Helmholtz equations is that the
operator $L()$ is linear. This property can be used to build better
estimates of a solution by successive *corrections* to an initial
guess. If we define an approximate solution $\tilde{a}$ as
$$
\tilde{a} + da = a
$$
where $a$ is the exact (unknown) solution, using the linearity of the
operator we find that $da$ verifies
$$
L(da) = b - L(\tilde{a})
$$
where the right-hand-side is often called the *residual* of the
approximate solution $\tilde{a}$.

## Multigrid cycle

Here we implement the multigrid cycle proper. Given an initial guess
*a*, a residual *res*, a correction field *da* and a relaxation
function *relax*, we will provide an improved guess at the end of the
cycle. */
#undef SEPS
#define SEPS 1e-12

#ifdef DEBUG_MODE_POISSON
  scalar residual_of_p[], divutmp[], divutmpAfter[];
#endif
bool relative_residual_poisson = false;
void mg_cycle (scalar * a, scalar * res, scalar * da,
	       void (* relax) (scalar * da, scalar * res,
			       int depth, void * data),
	       void * data,
	       int nrelax, int minlevel, int maxlevel)
{

  /**
  We first define the residual on all levels. */

  restriction (res);

  /**
  We then proceed from the coarsest grid (*minlevel*) down to the
  finest grid. */

  minlevel = min (minlevel, maxlevel);
  for (int l = minlevel; l <= maxlevel; l++) {

    /**
    On the coarsest grid, we take zero as initial guess. */

    if (l == minlevel)
      foreach_level_or_leaf (l)
        for (scalar s in da)
          s[] = 0.;

    /**
    On all other grids, we take as initial guess the approximate solution
    on the coarser grid bilinearly interpolated onto the current grid. */

    else
      foreach_level (l)
        for (scalar s in da)
          s[] = bilinear (point, s);

    /**
    We then apply homogeneous boundary conditions and do several
    iterations of the relaxation function to refine the initial guess. */

    boundary_level (da, l);
    for (int i = 0; i < nrelax; i++) {
      relax (da, res, l, data);
      boundary_level (da, l);
    }
  }

  /**
  And finally we apply the resulting correction to *a*. */

  foreach() {
    scalar s, ds;
    for (s, ds in a, da)
      s[] += ds[];
  }
  boundary (a);
}

/**
## Multigrid solver

The multigrid solver itself uses successive calls to the multigrid
cycle to refine an initial guess until a specified tolerance is
reached.

The maximum number of iterations is controlled by *NITERMAX* and the
tolerance by *TOLERANCE* with the default values below. */

int NITERMAX = 100, NITERMIN = 1;
double TOLERANCE = 1e-3;
double RELATIVE_RES_TOLERANCE = 0.1;
/**
Information about the convergence of the solver is returned in a structure. */

typedef struct {
  int i;              // number of iterations
  double resb, resa;  // maximum residual before and after the iterations
  double sum;         // sum of r.h.s.
  int nrelax;         // number of relaxations
  int minlevel;       // minimum level of the multigrid hierarchy
} mgstats;

/**
The user needs to provide a function which computes the residual field
(and returns its maximum) as well as the relaxation function. The
user-defined pointer *data* can be used to pass arguments to these
functions. The optional number of relaxations is *nrelax* (default is
one) and *res* is an optional list of fields used to store the
residuals. The minimum level of the hierarchy can be set (default is
zero i.e. the root cell). */

struct MGSolve {
  scalar * a, * b;
  double (* residual) (scalar * a, scalar * b, scalar * res,
		       void * data);
  void (* relax) (scalar * da, scalar * res, int depth,
		  void * data);
  void * data;

  int nrelax;
  scalar * res;
  int minlevel;
  double tolerance;
};

mgstats mg_solve (struct MGSolve p)
{

  /**
  We allocate a new correction and residual field for each of the scalars
  in *a*. */

  scalar * da = list_clone (p.a), * res = p.res;
  if (!res)
    for (scalar s in p.a) {
      scalar r = new scalar;
      res = list_append (res, r);
    }

  /**
  The boundary conditions for the correction fields are the
  *homogeneous* equivalent of the boundary conditions applied to
  *a*. */

  for (int b = 0; b < nboundary; b++)
    for (scalar s in da)
      s.boundary[b] = s.boundary_homogeneous[b];

  /**
  We initialise the structure storing convergence statistics. */

  mgstats s = {0};
  double sum = 0.;
  foreach (reduction(+:sum))
    for (scalar s in p.b)
      sum += s[];
  s.sum = sum;
  s.nrelax = p.nrelax > 0 ? p.nrelax : 4;

  /**
  Here we compute the initial residual field and its maximum. */

  double resb;
  resb = s.resb = s.resa = p.residual (p.a, p.b, res, p.data);

  /**
  We then iterate until convergence or until *NITERMAX* is reached. Note
  also that we force the solver to apply at least one cycle, even if the
  initial residual is lower than *TOLERANCE*. */
  #ifdef RELATIVE_RES
  	double res_previous1 = 0;
  	double res_previous2 = 0;
  #endif
  int patient = 0;
  if (p.tolerance == 0.)
    p.tolerance = TOLERANCE;
  for (s.i = 0;
       s.i < NITERMAX && (s.i < NITERMIN || s.resa > p.tolerance);
       s.i++) {
    mg_cycle (p.a, res, da, p.relax, p.data,
	      s.nrelax,
	      p.minlevel,
	      grid->maxdepth);
    s.resa = p.residual (p.a, p.b, res, p.data);

    /**
    We tune the number of relaxations so that the residual is reduced
    by between 2 and 20 for each cycle. This is particularly useful
    for stiff systems which may require a larger number of relaxations
    on the finest grid. */

#if 1
    if (s.resa > TOLERANCE) {
      if (resb/s.resa < 1.2 && s.nrelax < 100)
	    s.nrelax++;
      else if (resb/s.resa > 10 && s.nrelax > 2)
	    s.nrelax--;
    }
#else
    if (s.resa == resb) /* convergence has stopped!! */
      break;
    if (s.resa > resb/1.1 && p.minlevel < grid->maxdepth)
      p.minlevel++;
#endif

    resb = s.resa;
//break if resudual does not change. Weugene correction
#ifdef RELATIVE_RES
      double res1 = 0.5*(res_previous1+res_previous2);
      double res2 = 0.5*(res_previous1+s.resa);
	  double res_rel = fabs(res1 - res2)/(res2 + 1e-30);
      if (s.i == 2 && fabs(s.resa) < 1e-30) break;
      if( res_rel < RELATIVE_RES_TOLERANCE && patient > 3){
          scalar v = p.a[0];
          fprintf (ferr,
             "WARNING: Relative residual did not reach convergence for %s after %d iterations\n"
             "  rel_res: %g res: %g prev_res: %g %g sum: %g nrelax: %d\n",
             v.name, s.i, res_rel, s.resa, res_previous1, res_previous2, s.sum, s.nrelax), fflush (ferr);;
          break;
      }
      else{
          res_previous2 = res_previous1;
          res_previous1 = s.resa;
          patient++;
      }
#endif
  }
  s.minlevel = p.minlevel;

  /**
  If we have not satisfied the tolerance, we warn the user. */

  if (s.resa > p.tolerance) {
    scalar v = p.a[0];
    fprintf (ferr,
	     "WARNING: convergence for %s not reached after %d iterations\n"
	     "  res: %g sum: %g nrelax: %d\n", v.name,
	     s.i, s.resa, s.sum, s.nrelax), fflush (ferr);
  }

  /**
  We deallocate the residual and correction fields and free the lists. */

  if (!p.res)
    delete (res), free (res);
  delete (da), free (da);

  return s;
}

/**
## Application to the Poisson--Helmholtz equation

We now apply the generic multigrid solver to the Poisson--Helmholtz equation
$$
\nabla\cdot (\alpha\nabla a) + \lambda a = b
$$
We first setup the data structure required to pass the extra
parameters $\alpha$ and $\lambda$. We define $\alpha$ as a face
vector field because we need values at the face locations
corresponding to the face gradients of field $a$.

*alpha* and *lambda* are declared as *(const)* to indicate that the
function works also when *alpha* and *lambda* are constant vector
(resp. scalar) fields. If *tolerance* is set, it supersedes the
default *TOLERANCE* of the multigrid solver, *nrelax* controls the
initial number of relaxations (default is one), *minlevel* controls
the minimum level of the hierarchy (default is one) and *res* is an
optional list of fields used to store the final residual (which can be
useful to monitor convergence).

When using [embedded boundaries](embed.h) boundary fluxes on the
boundary need to be included. They are computed by the *embed_flux*
function. */

struct Poisson {
  scalar a, b;
  (const) face vector alpha;
  (const) scalar lambda;
  double tolerance;
  int nrelax, minlevel;
  scalar * res;
#if EMBED
  double (* embed_flux) (Point, scalar, vector, double *);
#endif
  double maxb;
};
//#if dimension == 1
////    #define refcoord (x==X0 + 0.5*Delta)
//    #define refcoord 1
//#elif dimension == 2
////    #define refcoord (x==X0 + 0.5*Delta && y==Y0 + 0.5*Delta)
//    #define refcoord y==Y0 + 0.5*Delta
//#else
////    #define refcoord (x==X0 + 0.5*Delta && y==Y0 + 0.5*Delta && z==Z0 + 0.5*Delta)
//    #define refcoord (y==(Y0 + 0.5*Delta) && (z==Z0 + 0.5*Delta))
//#endif
//bool reference_pressure = false;
//double xref, yref, zref, sref = 0;
//double first_value (scalar s, int l){
//    double scor = 0;
//    int masternode = 0;
//    foreach_boundary(left){
//        if (refcoord) {
//            scor = s[];
//#ifdef DEBUG_MODE_POISSON
//            if (scor>0.1) fprintf(ferr, "first value = %g; data=%g x = %g; y = %g; z = %g ", scor, data(0,0,0)[s.i], x, y, z);
//#endif
//            masternode = pid();
//            break;
//        }
//    }
//#if _MPI
//        MPI_Bcast (&scor, 1, MPI_DOUBLE, masternode, MPI_COMM_WORLD);
//#endif
////    foreach(){
////        if (refcoord) {
////            scor = s[];
////            fprintf(ferr, "first value = %g; x = %g; y = %g; z = %g ", scor, x, y, z);
////            break;
////        }
////    }
////    fprintf(ferr, " success \n");
//    return scor;
//}
//    xref = X0 + L0, yref = Y0 + 0.75*L0, zref = Z0;
//    double scor = interpolate (s, xref, yref) - sref;
//    fprintf(ferr, "first value = %g; x = %g; y = %g; z = %g\n", scor, xref, yref, zref);
//    return scor;
//    foreach_level(l) {
//        fprintf(ferr, "first value = %g; x = %g; y = %g\n", s[], x, y);
//        return s[];
//    }
//}

/**
We can now write the relaxation function. We first recover the extra
parameters from the data pointer. */

static void relax (scalar * al, scalar * bl, int l, void * data)
{
  scalar a = al[0], b = bl[0];
  struct Poisson * p = (struct Poisson *) data;
  (const) face vector alpha = p->alpha;
  (const) scalar lambda = p->lambda;

  /**
  We use either Jacobi (under)relaxation or we directly reuse values
  as soon as they are updated. For Jacobi, we need to allocate space
  for the new field *c*. Jacobi is useful mostly as it gives results
  which are independent of the order in which the cells are
  traversed. This is not the case for the simple traversal, which
  means for example that results will depend on whether a tree or
  a multigrid is used (because cells will be traversed in a different
  order). The same comment applies to OpenMP or MPI parallelism. In
  practice however Jacobi convergence tends to be slower than simple
  reuse. */

#if JACOBI
  scalar c[];
#else
  scalar c = a;
#endif

  /**
  We use the face values of $\alpha$ to weight the gradients of the
  5-points Laplacian operator. We get the relaxation function. */

  foreach_level_or_leaf (l) {
    double n = - sq(Delta)*b[], d = - lambda[]*sq(Delta);
    foreach_dimension() {
      n += alpha.x[1]*a[1] + alpha.x[]*a[-1];
      d += alpha.x[1] + alpha.x[];
    }
#if EMBED
    if (p->embed_flux) {
      double c, e = p->embed_flux (point, a, alpha, &c);
      n -= c*sq(Delta);
      d += e*sq(Delta);
    }
    if (!d)
      c[] = b[] = 0.;
    else
#endif // EMBED
      c[] = n/d;
  }
//  if (reference_pressure) {
//      double u_clip = first_value(a, l);
//      foreach_level_or_leaf(l)
//      {
//          c[] -= u_clip;
//      }
//  }

  /**
  For weighted Jacobi we under-relax with a weight of 2/3. */

#if JACOBI
  foreach_level_or_leaf (l)
    a[] = (a[] + 2.*c[])/3.;
#endif

#if TRASH
  scalar a1[];
  foreach_level_or_leaf (l)
    a1[] = a[];
  trash ({a});
  foreach_level_or_leaf (l)
    a[] = a1[];
#endif
}

/**
The equivalent residual function is obtained in a similar way in the
case of a Cartesian grid, however the case of the tree mesh
requires more careful consideration... */

static double residual (scalar * al, scalar * bl, scalar * resl, void * data)
{
  scalar a = al[0], b = bl[0], res = resl[0];
  struct Poisson * p = (struct Poisson *) data;
  (const) face vector alpha = p->alpha;
  (const) scalar lambda = p->lambda;
  double maxres = 0., maxb = p->maxb;
#if TREE
  /* conservative coarse/fine discretisation (2nd order) */
  face vector g[];
  foreach_face()
    g.x[] = alpha.x[]*face_gradient_x (a, 0);
  boundary_flux ({g});
  foreach (reduction(max:maxres)) {
    res[] = b[] - lambda[]*a[]; // lambda = 0 in usual case.
    foreach_dimension()
      res[] -= (g.x[1] - g.x[])/Delta;
#else // !TREE
  /* "naive" discretisation (only 1st order on trees) */
  foreach (reduction(max:maxres)) {
    res[] = b[] - lambda[]*a[];
    foreach_dimension()
      res[] += (alpha.x[0]*face_gradient_x (a, 0) -
		alpha.x[1]*face_gradient_x (a, 1))/Delta;
#endif // !TREE
#if EMBED
    if (p->embed_flux) {
      double c, e = p->embed_flux (point, a, alpha, &c);
      res[] += c - e*a[];
    }
#endif // EMBED
#ifdef DEBUG_MODE_POISSON
    residual_of_p[] = res[];
#endif
    if (fabs (res[]) > maxres)
      maxres = fabs (res[]);
  }
  boundary (resl);
  fprintf(ferr, "maxres*dt^2= %g maxres= %g maxb= %g maxres/maxb= %g\n", maxres*sq(dt), maxres, maxb, maxres/maxb);
  return maxres/maxb;
}

/**
## User interface

Finally we provide a generic user interface for a Poisson--Helmholtz
equation of the form
$$
\nabla\cdot (\alpha\nabla a) + \lambda a = b
$$ */

mgstats poisson (struct Poisson p)
{
	fprintf(ferr, "**********************\n");
  /**
  If $\alpha$ or $\lambda$ are not set, we replace them with constant
  unity vector (resp. zero scalar) fields. Note that the user is free to
  provide $\alpha$ and $\beta$ as constant fields. */

  if (!p.alpha.x.i)
    p.alpha = unityf;
  if (!p.lambda.i)
    p.lambda = zeroc;

  /**
  We need $\alpha$ and $\lambda$ on all levels of the grid. */

  face vector alpha = p.alpha;
  scalar lambda = p.lambda;
  restriction ({alpha,lambda});

  /**
  If *tolerance* is set it supersedes the default of the multigrid
  solver. */

  double defaultol = TOLERANCE;
  if (p.tolerance)
    TOLERANCE = p.tolerance;

  scalar a = p.a, b = p.b;
  if (relative_residual_poisson) {
    double maxb = 0;
    foreach(reduction(max:maxb)){
      if (fabs(b[]) > maxb) maxb = fabs(b[]);
    }
    if (maxb < SEPS) maxb = 1;
    p.maxb = maxb;
  }else{
    p.maxb = 1;
  }
#if EMBED
  if (!p.embed_flux && a.boundary[embed] != symmetry)
    p.embed_flux = embed_flux;
#endif // EMBED
  mgstats s = mg_solve ({a}, {b}, residual, relax,
			&p, p.nrelax, p.res, minlevel = max(1, p.minlevel));

  /**
  We restore the default. */

  if (p.tolerance)
    TOLERANCE = defaultol;

  return s;
}

/**
## Projection of a velocity field

The function below "projects" the velocity field *u* onto the space of
divergence-free velocity fields i.e.
$$
\mathbf{u}_f^{n+1} \leftarrow \mathbf{u}_f - \Delta t\alpha\nabla p
$$
so that
$$
\nabla\cdot\mathbf{u}_f^{n+1} = 0
$$
This gives the Poisson equation for the pressure
$$
\nabla\cdot(\alpha\nabla p) = \frac{\nabla\cdot\mathbf{u}_f}{\Delta t}
$$ */

struct Project {
  face vector uf;
  scalar p;
  face vector alpha; // optional: default unityf
  double dt;         // optional: default one
  int nrelax;        // optional: default four
  scalar fs;
  vector target_U;   // optional: default 0
  vector u;
};

trace
mgstats project (struct Project q)
{
    face vector uf = q.uf;
    scalar p = q.p;
    (const) face vector alpha = q.alpha.x.i ? q.alpha : unityf;
    double dt = q.dt ? q.dt : 1., d;
    int nrelax = q.nrelax ? q.nrelax : 4;

    /**
    We allocate a local scalar field and compute the divergence of
    $\mathbf{u}_f$. The divergence is scaled by *dt* so that the
    pressure has the correct dimension. */

    scalar div[];
    foreach() {
        d = 0.;
        foreach_dimension(){
            d += uf.x[1] - uf.x[];
        }
        div[] = d/(dt*Delta);
#ifdef DEBUG_MODE_POISSON
        divutmp[] = div[]*dt;
#endif
    }
    boundary((scalar*){div});
#ifdef DEBUG_MODE_POISSON
    boundary((scalar*){divutmp});
#endif
    /**
    We solve the Poisson problem. The tolerance (set with *TOLERANCE*) is
    the maximum relative change in volume of a cell (due to the divergence
    of the flow) during one timestep i.e. the non-dimensional quantity
    $$
    |\nabla\cdot\mathbf{u}_f|\Delta t
    $$
    Given the scaling of the divergence above, this gives */
// res=div(u + u*)/dt - laplace p
// res=div(u*)/dt - laplace delta p ~ dt
    mgstats mgp;
    if (relative_residual_poisson)
      mgp = poisson (p, div , alpha, tolerance = TOLERANCE, nrelax = nrelax);
    else
      mgp = poisson (p, div , alpha, tolerance = TOLERANCE/sq(dt), nrelax = nrelax);

//    double xref = X0 + L0 - L0/pow(2,maxlevel), yref = Y0 + 0.75*L0, zref = Z0; //reference location
////    double xref = -0.5, yref = Y0 + 0.5*L0, zref = Z0 + 0.5*L0; //reference location
//    double pref = 0;                                            //reference pressure
//    double pcor = interpolate (p, xref, yref, zref) - pref;
//    fprintf(ferr, "pcorr=%g at x=%g y=%g z=%g\n", pcor, xref, yref, zref);
//#if _MPI
//        MPI_Allreduce (MPI_IN_PLACE, &pcor, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
//#endif
//    foreach()
//        p[] -= pcor;
//    boundary ({p});
    /**
    And compute $\mathbf{u}_f^{n+1}$ using $\mathbf{u}_f$ and $p$. */

    foreach_face() uf.x[] -= dt*alpha.x[]*face_gradient_x (p, 0);
    boundary ((scalar *){uf});

#ifdef DEBUG_MODE_POISSON
    double divuf, maxdivuf = -1e30;
    foreach(reduction(max:maxdivuf)) {
      divuf = 0;
      foreach_dimension() divuf += (uf.x[1]-uf.x[])/Delta;
      divutmpAfter[] = divuf/dt;
      divuf = fabs(divuf);
      if (maxdivuf < divuf) maxdivuf = divuf;
    }
    boundary((scalar*){divutmpAfter});
    fprintf(ferr, "Projection MAX{div uf} = %g\n", maxdivuf);
#endif
    return mgp;
}

#endif
