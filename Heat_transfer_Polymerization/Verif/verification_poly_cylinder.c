#define BRINKMAN_PENALIZATION 1
#define DEBUG_MINMAXVALUES
#define DEBUG_BRINKMAN_PENALIZATION
#define DEBUG_MODE_POISSON
#define DEBUG_HEAT
#define DEBUG_MULTIGRID
#define HEAT_TRANSFER
#define REACTION_MODEL REACTION_MODEL_NON_AUTOCATALYTIC
#define VISCOSITY_MODEL VISCOSITY_MODEL_DUSI
#define DEBUG_OUTPUT_VTU_MPI
#define FILTERED
#define JACOBI 1

#define EPS_MAXA 2 // adapt based on Max-Min value
#define T_DIRICHLET_BC 1
// ignore solid and gas
#define mu(f, fs, alpha_doc, T) muf1(alpha_doc, T)


scalar mu_cell[];
scalar Phi_visc[], Phi_src[];
scalar temp_cyl[];
scalar levelset[];

static coord vel_s = {0, 0, 0};
//#include "curvature_partstr.h"
#include "centered-weugene.h"
#include "rheology_model.h"
#include "utils-weugene.h"
#include "output_htg.h"

int snapshot_i = 1000;
int iter_fp = 0;
double dt_vtk = 1;
double Uin, Tin, T_solid, Tam;
double RhoR, RhoRS, MuR, MuRS, CpR, CpRS, KappaR, KappaRS;
double Rho1, Rho2, Rho3;
double Mu0, Mu1, Mu2, Mu3;
double Kappa1, Kappa2, Kappa3;
double CP1, CP2, CP3;
double Ggrav, Ggrav_ndim;
double channel_diam, channel_length;
double Re; //Reynolds
double Fr; //Froude number Fr = sqrt(u^2/(g*cyl_diam))
int maxlevel = 9;
int minlevel = 5;
int LEVEL = 9;
double maxDT, maxDT0;
double mu_max = 0, nu_max = 0;
double feps = 1e-10, fseps = 1e-10, ueps = 1e-5, rhoeps = 1e-10, Teps = 1e-5, aeps = 1e-5, mueps=1e-5;
double mindelta, mindelta0;

char subname[150], logname[200];
char prefix[];


int main(int argc, char * argv[]) {
    fprintf(
        ferr,
        "./a.out T_solid, Tin, maxlevel, iter_fp, TOLERANCE_P, TOLERANCE_V, TOLERANCE_T, Htr, "
        "Arrhenius_const, Ea_by_R, subname\n"
    );
    TOLERANCE = 1e-9;
    NITERMIN = 1;
    NITERMAX = 100;
    CFL = 0.4;
    CFL_ARR = 0.5;
    DT = 1e-3;
    maxDT0 = 2.5e-3;

    N_smooth = 1; //three-phase-rheology.h

    stokes = true;
    stokes_heat = true;
    viscDissipation = false;

    m_bp = 1; // Brinkman layer resolution
    mbpT = 1;
    fprintf(ferr, "Stokes flow:stokes=%d, stokes_heat=%d\n", stokes, stokes_heat);
#if T_DIRICHLET_BC != 0
    fprintf(ferr, "T_DIRICHLET_BC: 1\n");
#endif
    sprintf(subname, "saturated_1");
    sprintf(logname, "log_saturated_1");
    // Physical parameters
	Uin = 1e-2;
    channel_diam = 0.2, channel_length = 20*channel_diam;
	Tin = 300, T_solid = 400, Tam = 300;
    // Gorthala, R., Roux, J. A., & Vaughan, J. G. (1994).
    // Resin Flow, Cure and Heat Transfer Analysis for Pultrusion Process.
    // Journal of Composite Materials, 28(6), 486–506.
    // doi:10.1177/002199839402800601
    // EPON 9310
	Htr = 355000; //J/kg
	Arrhenius_const = 80600;//1/s //originally in Gothala: 80600
	Ea_by_R = 64000/8.314;// Kelvin //origianally  64000/8.314;
    n_degree = 1.2;

	Eeta_by_Rg = 3.76e+4/8.314;// Kelvin Epon 9310,Safonov page 21
	chi = 20;
    Rho1 = 1200, Rho2 = 1.092, Rho3 = 1790;//air at 23 C Graphite
    CP1 = 1255, CP2 = 1006, CP3 = 712;//J/(kg*K)
//    Kappa1 = 0.2, Kappa2 = 0.02535, Kappa3 = 8.70;//W/(m*K)
    Kappa1 = 3012.05, Kappa2 = 381.78, Kappa3 = 3012.05;//W/(m*K)
    Ggrav = 0; // m/s^2

    if (argc > 1)
        T_solid = atof(argv[1]);
    if (argc > 2)
        Tin = atof(argv[2]);
    Mu0 = 3.85e-7, Mu1 = Mu0*exp(Eeta_by_Rg/Tin), Mu2 = 1.963e-5, Mu3 = Mu0*exp(Eeta_by_Rg/T_solid)*Rho3/Rho1;//air at 50C //Mu2 = 1.963e-5
	if (argc > 3)
        maxlevel = atoi(argv[3]);
    if (argc > 4)
        iter_fp = atoi(argv[4]);
	if (argc > 5)
        TOLERANCE_P = atof(argv[5]);
    if (argc > 6)
        TOLERANCE_V = atof(argv[6]);
    if (argc > 7)
        TOLERANCE_T = atof(argv[7]);
	if (argc > 8)
		Htr = atof(argv[8]);
	if (argc > 9)
		Arrhenius_const = atof(argv[9]);
	if (argc > 10)
		Ea_by_R = atof(argv[10]);
    if (argc > 11) {
        strcpy(subname, argv[11]);
        sprintf (logname, "log%s", subname);
    }
    if (argc > 12) {
        strcpy(prefix, argv[12]);
    }

	fprintf(
        ferr,
        "Props(SI): Mu0=%g, Mu1=%g, Mu2=%g, Mu3=%g, Rho1=%g, Rho2=%g,  Rho3=%g,\n"
        "           nu1=%g, nu2=%g, nu3=%g,\n"
        "           Kappa1=%g, Kappa2=%g, Kappa3=%g, CP1=%g, CP2=%g, CP3=%g,\n"
        "           Uin=%g, time*=%g, Tin=%g, T_solid=%g, Tam=%g,\n"
        "           Htr=%g, Arrenius=%g, Ea_by_R=%g, n_degree=%g,\n"
        "           Eeta_by_Rg=%g, chi=%g\n"
        "Geometry: channel_diam=%g,  domainSize=%g\n",
        Mu0, Mu1, Mu2, Mu3, Rho1, Rho2, Rho3,
        Mu1/Rho1, Mu2/Rho2, Mu3/Rho3,
        Kappa1, Kappa2, Kappa3, CP1, CP2, CP3,
        Uin, channel_diam/Uin, Tin, T_solid, Tam,
        Htr, Arrhenius_const, Ea_by_R, n_degree,
        Eeta_by_Rg, chi,
        channel_diam, channel_length
    );
    // Dimensionless numbers
	Re = Uin*channel_diam*Rho1/Mu1;
	Fr = sqrt(sq(Uin)/(Ggrav*channel_diam + SEPS));
    // Dimensionless parameters are chosen channel_diam, rho1, Cp1, Tin, Uin
    size(channel_length/channel_diam);
    mindelta = L0/pow(2, maxlevel);
    mindelta0 = L0/pow(2, 8);
    maxDT = maxDT0*sq(mindelta/mindelta0);  // h^2/DT = h0^2/DT0
	RhoR = Rho2/Rho1, RhoRS = Rho3/Rho1;
	MuR = Mu2/Mu1, MuRS = 1.0; // Mu3/Mu1; // Mu3/Mu1 = Rho3/Rho1 => RhoRS leads bad result?
	CpR = CP2/CP1, CpRS = CP3/CP1;
	KappaR = Kappa2/Kappa1, KappaRS = Kappa3/Kappa1;
    rho1 = 1; rho2 = RhoR; rho3 = RhoRS;
    mu0 = (1./Re)*(Mu0/Mu1); mu1 = (1./Re); mu2 = mu1*MuR; mu3 = mu1*MuRS; mu_eff = 0;
    // mu3 = mu1*(rho3/rho1)*sq(1.0/(m_bp*dx_min));
	Cp1 = 1; Cp2 = Cp1*CpR; Cp3 = Cp1*CpRS;//J/(kg*K)
	kappa1 = Kappa1/(Rho1*CP1*channel_diam*Uin + SEPS), kappa2 = kappa1*KappaR, kappa3 = kappa1*KappaRS;
    chi_conductivity = kappa1 / (rho1 * Cp1);
    Htr /= CP1*Tin;
	Arrhenius_const *= channel_diam/(Uin + SEPS);
	Ea_by_R /= Tin;
	Eeta_by_Rg /= Tin;
	Ggrav_ndim = 1./sq(Fr);
	Uin = 1;
    Tam /= Tin;
	T_solid /= Tin;
	Tin  /= Tin;
    channel_diam = 1;
    T_target = temp_cyl;
    const vector U_sol[] = {vel_s.x, vel_s.y, vel_s.z};
    target_U = U_sol;
    init_grid(1 << LEVEL);
    origin (-L0/2, -L0/2.);
    periodic(right);
    periodic(top);
	fprintf(
        ferr,
        "Dim-less vars: mu0=%g, mu1=%g, mu2=%g, mu3=%g,\n"
        "               rho1=%g, rho2=%g, rho3=%g,\n"
        "               nu1=%g, nu2=%g, nu3=%g,\n"
        "               kappa1=%g, kappa2=%g, kappa3=%g,\n"
        "               Cp1=%g, Cp2=%g, Cp3=%g,\n"
        "               RhoR=%g, RhoRS=%g, MuR=%g, MuRS=%g,\n"
        "               KappaR=%g, KappaRS=%g, CpR=%g, CpRS=%g\n"
        "               Uin=%g, Tin=%g, T_solid=%g, Tam=%g,\n"
        "               Htr=%g, Arrhenius_const=%g, Ea_by_R=%g, Eeta_by_Rg=%g,\n"
        "               L0=%g, channel_diam=%g, maxDT0=%g for maxlevel=8, maxDT=%g for maxlevel=%d\n"
        "               Ggrav_ndim=%g Uin=%g\n"
        "Dim-less nums: Re=%g,  Fr=%g\n"
        "Solver:        DTmax=%g, CFL=%g, CFL_ARR=%g, NITERMIN=%d,  NITERMAX=%d,\n"
        "               TOLERANCE_P=%g, TOLERANCE_V=%g, TOLERANCE_T=%g\n"
        "ADAPT:         minlevel=%d,  maxlevel=%d, feps=%g, fseps=%g, ueps=%g, Teps=%g, aeps=%g\n"
        "OUTPUT:        dt_vtk=%g,    number of procs=%d\n",
        mu0, mu1, mu2, mu3, rho1, rho2, rho3,
        mu1/rho1, mu2/rho2, mu3/rho3,
        kappa1, kappa2, kappa3,
        Cp1, Cp2, Cp3,
        RhoR, RhoRS, MuR, MuRS,
        KappaR, KappaRS, CpR, CpRS,
        Uin, Tin, T_solid, Tam,
        Htr, Arrhenius_const, Ea_by_R, Eeta_by_Rg,
        L0, channel_diam, maxDT0, maxDT, maxlevel,
        Ggrav_ndim, Uin,
        Re, Fr,
        DT, CFL, CFL_ARR, NITERMIN, NITERMAX,
        TOLERANCE_P, TOLERANCE_V, TOLERANCE_T,
        minlevel, maxlevel, feps, fseps, ueps, Teps, aeps,
        dt_vtk, npe()
    );

    fs.refine = fs.prolongation = fraction_refine;
    f.refine = f.prolongation = fraction_refine;
#ifdef _MPI
    int rank, psize, h_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    // get rank of this proces
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get total process number
    MPI_Comm_size(MPI_COMM_WORLD, &psize);
    MPI_Get_processor_name(hostname, &h_len);
    printf("rank:%d size: %d at %s h_len %d\n", rank, psize, hostname, h_len);
#endif
    for (maxlevel = 6; maxlevel < 10; maxlevel++){
        mindelta = L0/pow(2, maxlevel);
        mindelta0 = L0/pow(2, 8);
        DT *= sq(mindelta/mindelta0);  // h^2/DT = h0^2/DT0
        run();
    }
}

//#define T_BC min(T_solid + (Tin - T_solid)*tanh( 100*(sqrt(sq(x) + sq(y)) - channel_diam) ), T_solid)  #not compatible for convergence check
#define T_BC ((sqrt(sq(x) + sq(y)) <= 1.0) ? T_solid : (sqrt(sq(x) + sq(y)) <= 2.0) ? \
                                           (0.5*(Tin + T_solid) + 0.5*(T_solid - Tin)*cos(pi*(sqrt(sq(x) + sq(y)) - 1.0))) : \
                                           Tin)
u.n[left]  = dirichlet(0);
p[left]    = dirichlet(0);
pf[left]   = dirichlet(0);
f[left]    = dirichlet(1);
fs[left]    = dirichlet(0);
T[left]    = neumann(0);
alpha_doc[left] = neumann(0);

u.n[right] = dirichlet(0);
p[right]   = dirichlet(0);
pf[right]  = dirichlet(0);
T[right]    = neumann(0);
f[right]   = dirichlet(1);
fs[right]   = dirichlet(0);
alpha_doc[right] = neumann(0);

u.n[bottom] = dirichlet(0);
p[bottom]   = dirichlet(0);
pf[bottom]  = dirichlet(0);
T[bottom]    = neumann(0);
f[bottom]   = dirichlet(1);
fs[bottom]   = dirichlet(0);
alpha_doc[bottom] = neumann(0);

u.n[top] = dirichlet(0);
p[top]   = dirichlet(0);
pf[top]  = dirichlet(0);
T[top]    = neumann(0);
f[top]   = dirichlet(1);
fs[top]   = dirichlet(0);
alpha_doc[top] = neumann(0);

#define mynorm(v) (sqrt(sq(v.x) + sq(v.y)))
double xmin_center = 0, xmax_center = 0;

// Cylinder with diameter 1
double geometry(double x, double y, double z){
    return 1 - sqrt (sq (x) + sq (y));
}


void solid_func(scalar fs){
    vertex scalar phi[];
    foreach_vertex() {
        phi[] = geometry(x, y, z);
    }
    boundary ((scalar *){phi});
    fractions (phi, fs);
    boundary({fs});
}


event init (t = 0) {
    char name[300];
    sprintf (name, "restart_%s", subname);

    if (!restore (file = name)) {
        fprintf(ferr, "The file `%s` can not be successfully read! Initial conditions are set\n", name);
        int it = 0;
        scalar fs_smoothed[];
        do {
            solid_func(fs);
            filter_scalar(fs, fs_smoothed);
            foreach()
                T[] = T_BC;
            boundary({T});
        }while (adapt_wavelet({fs_smoothed, T}, (double []){feps, Teps}, maxlevel=maxlevel, minlevel=minlevel).nf != 0 && ++it <= 10);
        foreach() {
            f[] = 1;
            alpha_doc[] = 0;
            foreach_dimension() u.x[] = 0;
            temp_cyl[] = T_solid;
            levelset[] = geometry(x, y, z);
        }
        boundary({f, fs, alpha_doc, temp_cyl, levelset, u.x, u.y});
        update_T_target(levelset, T_target, cfl=0.5, nmax=100);
    }else{
        FILE *popen(const char *cmd_str, const char *mode);
        int pclose(FILE *stream);
        FILE *cmd;
        char result[5000];
        char cmd_str[200];
        strcpy(cmd_str, "grep \"vtk: iter_fp\" ");
        strcat(cmd_str, logname);
        strcat(cmd_str, " | awk \'{print $7}\' ");
        fprintf(ferr, "grep timesteps: %s", cmd_str);
        cmd = popen(cmd_str, "r");
        if (cmd == NULL) {
            fprintf(ferr, "Error in opening log file and searching timesteps");
            perror("popen");
            exit(EXIT_FAILURE);
        }
        int k = 0;
        while (fgets(result, sizeof(result), cmd)) {
            printf ("%s", result);
            // file_timesteps[k++] = atof(result);
        }
        cmd_str[0] = 0;
        strcpy(cmd_str, "grep \"vtk: iter_fp\" ");
        strcat(cmd_str, logname);
        strcat(cmd_str, " | tail -1 | awk \'{print $4}\' ");
        fprintf(ferr, "grep iter_fp: %s", cmd_str);
        cmd = popen(cmd_str, "r");
        // cmd = popen("grep \"vtk: iter_fp\" log | tail -1 | awk \'{print $4}\'", "r");
        if (cmd == NULL) {
            fprintf(ferr, "Error in opening log file and searching iter_fp");
            perror("popen");
            exit(EXIT_FAILURE);
        }
        fgets(result, sizeof(result), cmd);
        iter_fp = atoi(result) + 1;
        fprintf(ferr, "Read iter_fp+1: %d\n", iter_fp);
        pclose(cmd);
    }
}

event set_penalization(i++){
    double new_eta_s = 1e-5;
    double new_m_bp = 0;
    set_penalization_parameters (mu, rho, new_m_bp, new_eta_s);
    new_m_bp = 1;
    double new_eta_T = 0;
    double new_chi_conductivity = kappa1 / (rho1 * Cp1);
    set_heat_penalization_parameters(new_m_bp, new_eta_T, new_chi_conductivity);
}

event print_maxlevel (i++) {
    fprintf(ferr, "Maximum level of refinement at step %d: %d\n", i, grid->maxdepth);
}

void calc_mu(scalar mu_cell){
    foreach(){
        mu_cell[] = mu(f[], fs[], alpha_doc[], T[]);
    }
}

event properties(i++){
    calc_mu(mu_cell);
    if (viscDissipation)
        dissipation (Phi_visc, u, mu = mu);
    foreach(){
        Phi_src[] = f[]*(1 - fs[])*rho1*Htr*KT(T[])*FR(alpha_doc[]);
    }
    boundary({Phi_src});
}

event chem_conductivity_term (i++){
    foreach(){
        T_target[] = T[];
        if(fs[]!=1 && fs[]!=0){
            T_target[] = T_solid;
        }
    }
    boundary({T_target});
    int nmax = max(100 - i, 20);
    update_T_target(levelset, T_target, cfl=0.5, nmax=nmax);
}

event logoutput(t += 0.1){
    FILE *fp;
    const int Nvar = 4;
    const int Ninterp = 1001;
    static int firstWrite = 0;
    char name_vtu[1000];
    coord loc[Ninterp];
    double v[Nvar*Ninterp];
    for (int i = 0; i < Ninterp; i++){
        loc[i].x = 0.5*L0*i/(Ninterp - 1.0);
        loc[i].y = loc[i].z = 0;
    }
    sprintf(name_vtu, "cylinder_polymerization_basilisk_%s.csv", prefix);
    if (firstWrite == 0 && pid() == 0){
        fp = fopen(name_vtu, "w");
        fprintf (fp, "x,t,maxlevel,T,alpha,u,mu\n");
        fclose(fp);
        firstWrite++;
    }

    interpolate_array ((scalar*) {T, alpha_doc, u.x, mu_cell}, loc, Ninterp, v, true);
    if (pid() == 0){
        fp = fopen(name_vtu, "a");
        for (int i = 0; i < Ninterp; i++){
            int ii = i*Nvar;
            fprintf (fp, "%g,%g,%d,%g,%g,%g,%g\n", loc[i].x, t, maxlevel, v[ii], v[ii+1], v[ii+2], v[ii+3]);
        }
        fclose(fp);
    }
}

//#define VTK_SCALARS {rhov, T, T_target, alpha_doc, mu_cell, u.x, u.y, p, Phi_visc, Phi_src, uf.x}
//#define VTK_EPS_SCALARS {rhoeps, Teps, Teps, aeps, mueps, ueps, ueps, feps, feps, feps, ueps}
//event vtk_file (t += dt_vtk)
//{
//    char path[] = "res"; // no slash at the end!!
//    scalar l[];
//    foreach() {
//        l[] = level;
//    }
//
//    output_htg(
//        (scalar *){T, T_target, alpha_doc, p, fs, f, l, rhov, mu_cell, divutmp},
//        (vector *){u, g, uf, dbp, total_rhs, residual_of_u, divtauu, alpha, alphamv, kappa},
//        path, prefix, iter_fp, t
//    );
//    iter_fp++;
////    double eps_arr[] = VTK_EPS_SCALARS;
////    MinMaxValues((scalar *) VTK_SCALARS, eps_arr);
//}


/**
We adapt according to the error on the embedded geometry, velocity and
tracer fields. */
#define ADAPT_SCALARS {fs, T, alpha_doc}
#define ADAPT_EPS_SCALARS {fseps, Teps, aeps}

event adapt (i++){
	double eps_arr[] = ADAPT_EPS_SCALARS;
	MinMaxValues((scalar *) ADAPT_SCALARS, eps_arr);
	adapt_wavelet ((scalar *) ADAPT_SCALARS, eps_arr, maxlevel = maxlevel, minlevel = minlevel);
    foreach(){
        levelset[] = geometry(x, y, z);
    }
    boundary({levelset});
    if (i%100) count_cells(t, i);
}


event stop(t = 10);
