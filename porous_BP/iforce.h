/**
# Interfacial forces

We assume that the interfacial acceleration can be expressed as
$$
\phi\mathbf{n}\delta_s/\rho
$$
with $\mathbf{n}$ the interface normal, $\delta_s$ the interface Dirac
function, $\rho$ the density and $\phi$ a generic scalar field. Using
a CSF/Peskin-like approximation, this can be expressed as
$$
\phi\nabla f/\rho
$$
with $f$ the volume fraction field describing the interface.

The interfacial force potential $\phi$ is associated to each VOF
tracer. This is done easily by adding the following [field
attributes](/Basilisk C#field-attributes). */

attribute {
  scalar phi;
}

/**
Interfacial forces are a source term in the right-hand-side of the
evolution equation for the velocity of the [centered Navier--Stokes
solver](navier-stokes/centered.h) i.e. it is an acceleration. If
necessary, we allocate a new vector field to store it. */

event defaults (i = 0) {  
  if (is_constant(a.x)) {
    a = new face vector;
    foreach_face()
      a.x[] = 0.;
    boundary ((scalar *){a});
  }
}

/**
The calculation of the acceleration is done by this event, overloaded
from [its definition](navier-stokes/centered.h#acceleration-term) in
the centered Navier--Stokes solver. */

event acceleration (i++)
{
  
  /**
  We check for all VOF interfaces for which $\phi$ is allocated. The
  corresponding volume fraction fields will be stored in *list*. */
	fprintf(ferr, "accel0\n");
  scalar * list = NULL;
  for (scalar f in interfaces)
    if (f.phi.i) {
      list = list_add (list, f);

      /**
      To avoid undeterminations due to round-off errors, we remove
      values of the volume fraction larger than one or smaller than
      zero. */

      foreach()
	f[] = clamp (f[], 0., 1.);
      boundary ({f});
    }

  /**
  On trees we need to make sure that the volume fraction gradient
  is computed exactly like the pressure gradient. This is necessary to
  ensure well-balancing of the pressure gradient and interfacial force
  term. To do so, we apply the same prolongation to the volume
  fraction field as applied to the pressure field. */
	fprintf(ferr, "accel1\n");
#if TREE
  for (scalar f in list)
    f.prolongation = p.prolongation;
  boundary (list);
#endif
	fprintf(ferr, "accel2\n");
  /**
  Finally, for each interface for which $\phi$ is allocated, we
  compute the interfacial force acceleration
  $$
  \phi\mathbf{n}\delta_s/\rho \approx \alpha\phi\nabla f
  $$ 
  */
	fprintf(ferr, "accel3\n");
  face vector ia = a;
  foreach_face()
    for (scalar f in list)
      if (f[] != f[-1]) {

	/**
	We need to compute the potential *phif* on the face, using its
	values at the center of the cell. If both potentials are
	defined, we take the average, otherwise we take a single
	value. If all fails we set the potential to zero: this should
	happen only because of very pathological cases e.g. weird
	boundary conditions for the volume fraction. */
	
	scalar phi = f.phi;
	double phif =
	  (phi[] < nodata && phi[-1] < nodata) ?
	  (phi[] + phi[-1])/2. :
	  phi[] < nodata ? phi[] :
	  phi[-1] < nodata ? phi[-1] :
	  0.;
	
	ia.x[] += alpha.x[]/fm.x[]*phif*(f[] - f[-1])/Delta;
      }
	fprintf(ferr, "accel4\n");
  /**
  On trees, we need to restore the prolongation values for the
  volume fraction field. */
  
#if TREE
  for (scalar f in list)
    f.prolongation = fraction_refine;
  boundary (list);
#endif
	fprintf(ferr, "accel5\n");
  /**
  Finally we free the potential fields and the list of volume
  fractions. */

  for (scalar f in list) {
    scalar phi = f.phi;
    delete ({phi});
    f.phi.i = 0;
  }
  free (list);
}

/**
## References

See Section 3, pages 8-9 of:

~~~bib
@Article{popinet2018,
  author =  {S. Popinet},
  title =   {Numerical models of surface tension},
  journal = {Annual Review of Fluid Mechanics},
  pages =   {1--28},
  volume =  {50},
  year =    {2018},
  doi =     {10.1146/annurev-fluid-122316-045034},
  url =     {https://hal.archives-ouvertes.fr/hal-01528255}
}
~~~
*/
