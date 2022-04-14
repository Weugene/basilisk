/*
This function writes one XML file which allows to read the *.vtu files generated
by output_vtu_ascii_foreach() when used in MPI. Tested in (quad- and oct-)trees
using MPI.
*/
#define SMALL_VAL 0
//#define SMALL_VAL 1e-12

//#define MY_BOX_CONDITION true
#ifdef _MPI
#if dimension == 1
    #define MY_BOX_CONDITION ((!Period.x) || ((x - Pmin.x - 0.5*Delta > 0) && (Pmax.x - x - 0.5*Delta > 0)))
#elif dimension == 2
	#define MY_BOX_CONDITION (((!Period.x) || ((x - Pmin.x - 0.5*Delta > 0) && (Pmax.x - x - 0.5*Delta > 0))) && ((!Period.y) || ((y - Pmin.y - 0.5*Delta > 0) && (Pmax.y - y - 0.5*Delta > 0))))
#elif dimension > 2
    #define MY_BOX_CONDITION (((!Period.x) || ((x - Pmin.x - 0.5*Delta > 0) && (Pmax.x - x - 0.5*Delta > 0))) && ((!Period.y) || ((y - Pmin.y - 0.5*Delta > 0) && (Pmax.y - y - 0.5*Delta > 0))) && ((!Period.z) || ((z - Pmin.z - 0.5*Delta > 0) && (Pmax.z - z - 0.5*Delta > 0))))
#endif
#else
#if dimension == 1
    #define MY_BOX_CONDITION ((!Period.x) || (Pmax.x - x - 0.5*Delta > 0))
#elif dimension == 2
    #define MY_BOX_CONDITION (((!Period.x) || (Pmax.x - x - 0.5*Delta > 0)) && ((!Period.y) || (Pmax.y - y - 0.5*Delta > 0)))
#elif dimension > 2
    #define MY_BOX_CONDITION (((!Period.x) || (Pmax.x - x - 0.5*Delta > 0)) && ((!Period.y) || (Pmax.y - y - 0.5*Delta > 0)) && ((!Period.z) || (Pmax.z - z - 0.5*Delta > 0)))
#endif
#endif

#ifdef PRINT_ALL_VALUES
#if dimension == 1
    #define LISTDIM 3
    #define FVLISTDIM 2
#elif dimension == 2
    #define LISTDIM 5
    #define FVLISTDIM 8
#elif dimension > 2
    #define LISTDIM 7
    #define FVLISTDIM 18
#endif
#else
    #define LISTDIM 1
    #define FVLISTDIM 8
#endif

#if dimension == 1
    char components_name[] = "ComponentName0=\"left_X\" ComponentName1=\"right_X\" ";
#elif dimension == 2
    char components_name[] = "ComponentName0=\"left_X\" ComponentName1=\"left_Y\" "
                             "ComponentName2=\"right_X\" ComponentName3=\"right_Y\" "
                             "ComponentName4=\"bottom_X\" ComponentName5=\"bottom_Y\" "
                             "ComponentName6=\"top_X\" ComponentName7=\"top_Y\" ";
#elif dimension > 2
    char components_name[] = "ComponentName0=\"left_X\" ComponentName1=\"left_Y\" ComponentName2=\"left_Z\" "
                             "ComponentName3=\"right_X\" ComponentName4=\"right_Y\" ComponentName5=\"right_Z\" "
                             "ComponentName6=\"bottom_X\" ComponentName7=\"bottom_Y\" ComponentName8=\"bottom_Z\" "
                             "ComponentName9=\"top_X\" ComponentName10=\"top_Y\" ComponentName11=\"top_Z\" "
                             "ComponentName12=\"back_X\" ComponentName13=\"back_Y\" ComponentName14=\"back_Z\" "
                             "ComponentName15=\"front_X\" ComponentName16=\"front_Y\" ComponentName17=\"fronts_Z\" ";
#endif
#define MAX_STRING_SIZE 40
char array_subname[][MAX_STRING_SIZE] = { "left", "center", "right",
                                          "bottom", "top",
                                          "back", "front"
                                        };
int aid[7][3] = { {-1,0,0}, {0,0,0}, {1,0,0},
                  {0,-1,0}, {0,1,0},
                  {0,0,-1}, {0,0,1}
                };

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
void output_pvtu_ascii (scalar * list, vector * vlist, vector * fvlist, int n, FILE * fp, char * subname)
{
    fputs ("<?xml version=\"1.0\"?>\n"
    "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
    fputs ("\t <PUnstructuredGrid GhostLevel=\"0\">\n", fp);
    fputs ("\t\t\t <PCellData Scalars=\"scalars\">\n", fp);
#ifndef PRINT_ALL_VALUES
    for (scalar s in list) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", s.name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    for (vector v in vlist) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"ascii\">\n", v.x.name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
#else
    for (scalar s in list) {
      for (long long int i=0; i<LISTDIM; i++)
            fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s_%s\" format=\"ascii\">\n", s.name, array_subname[i]);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    for (vector v in vlist) {
        for (long long int i=0; i<LISTDIM; i++)
            fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"%d\" Name=\"%s_%s\" format=\"ascii\">\n", 3, v.x.name, array_subname[i]);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
#endif
    for (vector v in fvlist) {
        fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"%d\" %s Name=\"%s\" format=\"ascii\">\n", FVLISTDIM, components_name, v.x.name);
        fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    fputs ("\t\t\t </PCellData>\n", fp);
    fputs ("\t\t\t <PPoints>\n", fp);
    fputs ("\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);
    fputs ("\t\t\t\t </PDataArray>\n", fp);
    fputs ("\t\t\t </PPoints>\n", fp);

    for (int i = 0; i < npe(); i++)
      fprintf (fp, "<Piece Source=\"%s_n%3.3d.vtu\"/> \n", subname, i);

    fputs ("\t </PUnstructuredGrid>\n", fp);
    fputs ("</VTKFile>\n", fp);
}

/*
This function writes one XML VTK file per PID process of type unstructured grid
(*.vtu) which can be read using Paraview. File stores scalar and vector fields
defined at the center points. Results are recorded on ASCII format. If one writes
one *.vtu file per PID process this function may be combined with
output_pvtu_ascii() above to read in parallel. Tested in (quad- and oct-)trees
using MPI. Also works with solids (when not using MPI).
*/
void output_vtu_ascii_foreach (scalar * list, vector * vlist, vector * fvlist, int n, FILE * fp, bool linear)
{
    coord Pmin = {X0 + SMALL_VAL, Y0 + SMALL_VAL, Z0 + SMALL_VAL};
    coord Pmax = {X0 + L0 - SMALL_VAL, Y0 + L0 - SMALL_VAL, Z0 + L0 - SMALL_VAL};
#if defined(_OPENMP)
  long long int num_omp = omp_get_max_threads();
  omp_set_num_threads(1);
#endif

  vertex scalar marker[];
  long long int no_points = 0, no_cells=0 ;
  foreach_vertex(){
    if (MY_BOX_CONDITION) {
      marker[] = no_points++;
    }else{
	  marker[] = -1;
    }
  }
  foreach(){
    if (MY_BOX_CONDITION) no_cells += 1;
  }

  fputs ("<?xml version=\"1.0\"?>\n"
  "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
  fputs ("\t <UnstructuredGrid>\n", fp);
  fprintf (fp,"\t\t <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n", no_points, no_cells);
  fputs ("\t\t\t <CellData Scalars=\"scalars\">\n", fp);
#ifndef PRINT_ALL_VALUES
  for (scalar s in list) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", s.name);
    foreach(){
      if (MY_BOX_CONDITION)
        fprintf (fp, "\t\t\t\t\t %g\n", val(s));
    }
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  for (vector v in vlist) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"ascii\">\n", v.x.name);
    foreach(){
      if (MY_BOX_CONDITION){
#if dimension == 1
          fprintf (fp, "\t\t\t\t\t %g 0 0.\n", val(v.x));
#endif
#if dimension == 2
          fprintf (fp, "\t\t\t\t\t %g %g 0.\n", val(v.x), val(v.y));
#endif
#if dimension > 2
          fprintf (fp, "\t\t\t\t\t %g %g %g\n", val(v.x), val(v.y), val(v.z));
#endif
      }
    }
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
#else
  for (scalar s in list) {
      for (long long int i=0; i<LISTDIM; i++){
          fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s_%s\" format=\"ascii\">\n", s.name, array_subname[i]);
          foreach(){
              if (MY_BOX_CONDITION)
                fprintf (fp, "\t\t\t\t\t %g \n",s[aid[i][0], aid[i][1], aid[i][2]]);
          }
      }
      fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  for (vector v in vlist) {
      for (long long int i=0; i<LISTDIM; i++){
          fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"%d\" Name=\"%s_%s\" format=\"ascii\">\n", 3, v.x.name, array_subname[i]);
          foreach(){
              if (MY_BOX_CONDITION){
                  #if dimension == 1
                    fprintf (fp, "\t\t\t\t\t %g 0 0.\n", v.x[aid[i][0],aid[i][1],aid[i][2]]);
                  #endif
                  #if dimension == 2
                    fprintf (fp, "\t\t\t\t\t %g %g 0.\n", v.x[aid[i][0],aid[i][1],aid[i][2]], v.y[aid[i][0],aid[i][1],aid[i][2]]);
                  #endif
                  #if dimension > 2
                    fprintf (fp, "\t\t\t\t\t %g %g %g\n", v.x[aid[i][0],aid[i][1],aid[i][2]], v.y[aid[i][0],aid[i][1],aid[i][2]], v.z[aid[i][0],aid[i][1],aid[i][2]]);
                  #endif
              }
          }
    }
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
#endif
  for (vector v in fvlist) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"%d\" %s Name=\"%s\" format=\"ascii\">\n", FVLISTDIM, components_name, v.x.name);
    foreach(){
      if (MY_BOX_CONDITION){
#if dimension == 1
        double arr[FVLISTDIM]={v.x[], v.x[1]};
#endif
#if dimension == 2
        double arr[FVLISTDIM]={v.x[0,0], v.y[-1,0],
                                     v.x[1,0], v.y[1,0],
                                     v.x[0,-1], v.y[0,0],
                                     v.x[0,1], v.y[0,1]};
#endif
#if dimension > 2
        double arr[FVLISTDIM]={v.x[0,0], v.y[-1,0], v.z[-1,0],
                                     v.x[1,0], v.y[1,0], v.z[1,0],
                                     v.x[0,-1], v.y[0,0], v.z[0,-1],
                                     v.x[0,1], v.y[0,1], v.z[0,1],
                                     v.x[0,0,-1], v.y[0,0,-1], v.z[0,0,0],
                                     v.x[0,0,1], v.y[0,0,1], v.z[0,0,1]};
#endif
        fprintf (fp, "\t\t\t\t\t ");
        for (int i = 0; i < FVLISTDIM; i++){
            fprintf (fp, "%g ", arr[i]);
        }
        fprintf (fp, "\n");
      }
    }
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }

  fputs ("\t\t\t </CellData>\n", fp);
  fputs ("\t\t\t <Points>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);
  foreach_vertex(){
    if (MY_BOX_CONDITION)
        fprintf (fp, "\t\t\t\t\t %g %g %g\n", x, y, z);
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Points>\n", fp);
  fputs ("\t\t\t <Cells>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n", fp);
  foreach(){
    if (MY_BOX_CONDITION)
    #if dimension == 1
      fprintf (fp, "\t\t\t\t\t %d %d \n", (int)marker[], (int)marker[1]);
    #endif
    #if dimension == 2
      fprintf (fp, "\t\t\t\t\t %d %d %d %d \n", (int)marker[], (int)marker[1,0], (int)marker[1,1], (int)marker[0,1]);
    #endif
    #if dimension > 2
      fprintf (fp, "\t\t\t\t\t %d %d %d %d %d %d %d %d \n", (int)marker[], (int)marker[1,0,0], (int)marker[1,1,0], (int)marker[0,1,0], (int)marker[0,0,1], (int)marker[1,0,1], (int)marker[1,1,1], (int)marker[0,1,1]);
    #endif
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n", fp);

  for (long long int i = 1; i < no_cells+1; i++){
#if dimension == 2
    fprintf (fp, "\t\t\t\t\t %lld \n", i*4);
#endif
#if dimension > 2
    fprintf (fp, "\t\t\t\t\t %lld \n", i*8);
#endif
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n", fp);
  foreach(){
    if (MY_BOX_CONDITION)
    #if dimension == 1
      fputs ("\t\t\t\t\t 3 \n", fp); //VTK_LINE (=3)
    #endif
    #if dimension == 2
      fputs ("\t\t\t\t\t 9 \n", fp); //VTK_QUAD (=9)
    #endif
    #if dimension > 2
      fputs ("\t\t\t\t\t 12 \n", fp); //VTK_HEXAHEDRON
    #endif
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Cells>\n", fp);
  fputs ("\t\t </Piece>\n", fp);
  fputs ("\t </UnstructuredGrid>\n", fp);
  fputs ("</VTKFile>\n", fp);
  fflush (fp);
#if defined(_OPENMP)
  omp_set_num_threads(num_omp);
#endif
}

/*
This function writes one XML file which allows to read the *.vtu files generated
by output_vtu_bin_foreach() when used in MPI. Tested in (quad- and oct-)trees
using MPI.
*/
void output_pvtu_bin (scalar * list, vector * vlist, vector * fvlist, int n, FILE * fp, char * subname)
{
    fputs ("<?xml version=\"1.0\"?>\n"
    "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
    fputs ("\t <PUnstructuredGrid GhostLevel=\"0\">\n", fp);
    fputs ("\t\t\t <PCellData Scalars=\"scalars\">\n", fp);
#ifndef PRINT_ALL_VALUES
    for (scalar s in list) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s\" format=\"appended\"/>\n", s.name);
    }
    for (vector v in vlist) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"%d\" Name=\"%s\" format=\"appended\"/>\n", 3, v.x.name);
    }
#else
    for (scalar s in list) {
        for (long long int i=0; i<LISTDIM; i++)
            fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s_%s\" format=\"appended\"/>\n", s.name, array_subname[i]);
    }
    for (vector v in vlist) {
        for (long long int i=0; i<LISTDIM; i++)
            fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"%d\" Name=\"%s_%s\" format=\"appended\"/>\n", 3, v.x.name, array_subname[i]);
    }
#endif
    for (vector v in fvlist) {
        fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"%d\" %s Name=\"%s\" format=\"appended\"/>\n", FVLISTDIM, components_name, v.x.name);
    }
    fputs ("\t\t\t </PCellData>\n", fp);
    fputs ("\t\t\t <PPoints>\n", fp);
    fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"%d\" format=\"ascii\"/>\n", 3);
    fputs ("\t\t\t </PPoints>\n", fp);

    for (int i = 0; i < npe(); i++)
      fprintf (fp, "<Piece Source=\"%s_n%3.3d.vtu\"/> \n", subname, i);

    fputs ("\t </PUnstructuredGrid>\n", fp);
    fputs ("</VTKFile>\n", fp);
}

/*
This function writes one XML VTK file per PID process of type unstructured grid
(*.vtu) which can be read using Paraview. File stores scalar and vector fields
defined at the center points. Results are recorded on binary format. If one writes
one *.vtu file per PID process this function may be combined with
output_pvtu_bin() above to read in parallel. Tested in (quad- and oct-)trees
using MPI. Also works with solids (when not using MPI).
*/

void output_vtu_bin_foreach (scalar * list, vector * vlist, vector * fvlist, int n, FILE * fp, bool linear)
{
  coord Pmin = {X0 + SMALL_VAL, Y0 + SMALL_VAL, Z0 + SMALL_VAL};
  coord Pmax = {X0 + L0 - SMALL_VAL, Y0 + L0 - SMALL_VAL, Z0 + L0 - SMALL_VAL};
#if defined(_OPENMP)
  int num_omp = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
  vertex scalar marker[];
  long long int no_points = 0, no_cells = 0;
  foreach_vertex(){
    if (MY_BOX_CONDITION) {
      marker[] = no_points++;// locally for each CPU counting the number of points
    }else{
    	marker[] = -1; //if you see -1 in vtu file=> there is a mistake
    	fprintf(ferr, "marker[] = -1, x=%g y=%g\n", x, y);
    }
  }
  foreach(){
    if (MY_BOX_CONDITION) no_cells++;// locally for each CPU counting the number of cells
  }
  fputs ("<?xml version=\"1.0\"?>\n"
  "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
  fputs ("\t <UnstructuredGrid>\n", fp);
  fprintf (fp,"\t\t <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n", no_points, no_cells);
  fputs ("\t\t\t <CellData Scalars=\"scalars\">\n", fp);
  long long int count = 0;
#ifndef PRINT_ALL_VALUES
  for (scalar s in list) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"%lld\">\n", s.name, count);
    count += ((no_cells)+1)*8;
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  for (vector v in vlist) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"%d\"  format=\"appended\" offset=\"%lld\">\n", v.x.name, 3, count);
    count += (no_cells*3+1)*8;
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
#else
  for (scalar s in list) {
      for (long long int i=0; i<LISTDIM; i++){
        fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s_%s\" format=\"appended\" offset=\"%lld\">\n", s.name, array_subname[i], count);
        count += ((no_cells)+1)*8;
        fputs ("\t\t\t\t </DataArray>\n", fp);
      }
  }
  for (vector v in vlist) {
      for (long long int i=0; i<LISTDIM; i++){
        fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s_%s\" NumberOfComponents=\"%d\"  format=\"appended\" offset=\"%lld\">\n", v.x.name, array_subname[i], 3, count);
        count += (no_cells*3+1)*8;
        fputs ("\t\t\t\t </DataArray>\n", fp);
      }
  }
#endif
  for (vector v in fvlist) {
    fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"%d\"  %s format=\"appended\" offset=\"%lld\">\n", v.x.name, FVLISTDIM, components_name, count);
    count += (no_cells*FVLISTDIM+1)*8;
    fputs ("\t\t\t\t </DataArray>\n", fp);
  }
  fputs ("\t\t\t </CellData>\n", fp);
  fputs ("\t\t\t <Points>\n", fp);
  fprintf (fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"%d\"  format=\"appended\" offset=\"%lld\">\n", 3, count);
  count += (no_points*3+1)*8;
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Points>\n", fp);
  fputs ("\t\t\t <Cells>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n", fp);
  foreach(){
    if (MY_BOX_CONDITION) {
#if dimension == 1
	    fprintf (fp, "\t\t\t\t\t %d %d \n", (int)marker[], (int)marker[1]);
#endif
#if dimension == 2
	    fprintf (fp, "\t\t\t\t\t %d %d %d %d \n", (int)marker[], (int)marker[1,0], (int)marker[1,1], (int)marker[0,1]);
#endif
#if dimension > 2
	    fprintf (fp, "\t\t\t\t\t %d %d %d %d %d %d %d %d \n", (int)marker[], (int)marker[1,0,0], (int)marker[1,1,0], (int)marker[0,1,0], (int)marker[0,0,1], (int)marker[1,0,1], (int)marker[1,1,1], (int)marker[0,1,1]);
#endif
    }
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n", fp);
  for (long long int i = 1; i < no_cells+1; i++){
#if dimension == 1
    fprintf (fp, "%lld \n", i*2);
#endif
#if dimension == 2
    fprintf (fp, "%lld \n", i*4);
#endif
#if dimension > 2
    fprintf (fp, "%lld \n", i*8);
#endif
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t\t <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n", fp);
  foreach(){
    if (MY_BOX_CONDITION)
    #if dimension == 1
      fputs ("3 \n", fp); //VTK_LINE (=3)
    #endif
    #if dimension == 2
      fputs ("9 \n", fp); //VTK_QUAD (=9)
    #endif
    #if dimension > 2
      fputs ("12 \n", fp); //VTK_HEXAHEDRON
    #endif
  }
  fputs ("\t\t\t\t </DataArray>\n", fp);
  fputs ("\t\t\t </Cells>\n", fp);
  fputs ("\t\t </Piece>\n", fp);
  fputs ("\t </UnstructuredGrid>\n", fp);
  fputs ("\t <AppendedData encoding=\"raw\">\n", fp);
  fputs ("_", fp);
  unsigned long long block_len=no_cells*8;
#if dimension == 1
  double y=0, vy=0;
  double z=0, vz=0;
#endif
#if dimension == 2
  double z=0, vz=0;
#endif
#ifndef PRINT_ALL_VALUES
  for (scalar s in list) {
    fwrite (&block_len, sizeof (unsigned long long), 1, fp);
    foreach()
      if (MY_BOX_CONDITION)
        fwrite (&val(s), sizeof (double), 1, fp);
  }
  block_len=no_cells*8*3;
  for (vector v in vlist) {
    fwrite (&block_len, sizeof (unsigned long long), 1, fp);
    foreach(){
      if (MY_BOX_CONDITION){
      #if dimension == 1
        fwrite (&val(v.x), sizeof (double), 1, fp);
        fwrite (&vy, sizeof (double), 1, fp);
        fwrite (&vz, sizeof (double), 1, fp);
      #endif
      #if dimension == 2
        fwrite (&val(v.x), sizeof (double), 1, fp);
        fwrite (&val(v.y), sizeof (double), 1, fp);
        fwrite (&vz, sizeof (double), 1, fp);
      #endif
      #if dimension > 2
        fwrite (&val(v.x), sizeof (double), 1, fp);
        fwrite (&val(v.y), sizeof (double), 1, fp);
        fwrite (&val(v.z), sizeof (double), 1, fp);
      #endif
      }
    }
  }
#else
  for (scalar s in list) {
      for (long long int i=0; i<LISTDIM; i++){
          fwrite (&block_len, sizeof (unsigned long long), 1, fp);
          foreach()
            if (MY_BOX_CONDITION)
                fwrite (&val(s, aid[i][0], aid[i][1], aid[i][2]), sizeof (double), 1, fp);
    }
  }
  block_len=no_cells*8*3;
  for (vector v in vlist) {
      for (long long int i=0; i<LISTDIM; i++){
        long long int ai=aid[i][0], aj=aid[i][1], ak=aid[i][2];
        fwrite (&block_len, sizeof (unsigned long long), 1, fp);
        foreach(){
          if (MY_BOX_CONDITION){
          #if dimension == 1
            fwrite (&val(v.x, ai, aj, ak), sizeof (double), 1, fp);
            fwrite (&vy, sizeof (double), 1, fp);
            fwrite (&vz, sizeof (double), 1, fp);
          #endif
          #if dimension == 2
            fwrite (&val(v.x, ai, aj, ak), sizeof (double), 1, fp);
            fwrite (&val(v.y, ai, aj, ak), sizeof (double), 1, fp);
            fwrite (&vz, sizeof (double), 1, fp);
          #endif
          #if dimension > 2
            fwrite (&val(v.x, ai, aj, ak), sizeof (double), 1, fp);
            fwrite (&val(v.y, ai, aj, ak), sizeof (double), 1, fp);
            fwrite (&val(v.z, ai, aj, ak), sizeof (double), 1, fp);
          #endif
          }
        }
    }
  }
#endif
  block_len=no_cells*8*FVLISTDIM;
  for (vector v in fvlist) {
    fwrite (&block_len, sizeof (unsigned long long), 1, fp);
    foreach(){
      if (MY_BOX_CONDITION){
#if dimension == 1
        double arr[FVLISTDIM]={v.x[], v.x[1]};
#endif
#if dimension == 2
//        double arr[FVLISTDIM]={  v.x[0,1], v.x[0,0], v.x[0,-1], v.x[1,1], v.x[1,0], v.x[1,-1],
//                                 v.y[-1,0], v.y[0,0], v.y[1,0], v.y[-1,1], v.y[0,1], v.y[1,1]};
          double arr[FVLISTDIM]={  v.x[0,0], v.y[-1,0],
                                   v.x[1,0], v.y[1,0],
                                   v.x[0,-1], v.y[0,0],
                                   v.x[0,1], v.y[0,1]
                                };
#endif
#if dimension > 2
          double arr[FVLISTDIM]={  v.x[0,0], v.y[-1,0], v.z[-1,0],
                                   v.x[1,0], v.y[1,0], v.z[1,0],
                                   v.x[0,-1], v.y[0,0], v.z[0,-1],
                                   v.x[0,1], v.y[0,1], v.z[0,1],
                                   v.x[0,0,-1], v.y[0,0,-1], v.z[0,0,0],
                                   v.x[0,0,1], v.y[0,0,1], v.z[0,0,1]
                                };
#endif
        fwrite (&arr, FVLISTDIM*sizeof(double), 1, fp);
      }
    }
  }
  block_len=no_points*8*3;
  fwrite (&block_len, sizeof (unsigned long long), 1, fp);
  foreach_vertex(){
    if (MY_BOX_CONDITION){
      fwrite (&x, sizeof (double), 1, fp);
      fwrite (&y, sizeof (double), 1, fp);
      fwrite (&z, sizeof (double), 1, fp);
    }
  }
  fputs ("\t\n", fp);
  fputs ("\t </AppendedData>\n", fp);
  fputs ("</VTKFile>\n", fp);
  fflush (fp);
#if defined(_OPENMP)
  omp_set_num_threads(num_omp);
#endif
}

void output_pvd_file(FILE * fp, int nf, float * file_timesteps, char * subname){
//    fputs("<?xml version=\"1.0\"?>\n",fp);
    fputs ("<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n"
           "\t <Collection>\n", fp);
    for (int i=0; i<=nf; i++) {
        fprintf(fp, "\t\t<DataSet timestep=\"%g\" part=\"0\" file=\"res/%s_0_%4.4d.pvtu\"/>\n", file_timesteps[i], subname, i);
    }//12.9g
    fputs ("\t </Collection>\n "
           "</VTKFile>\n", fp);
}
/* output_vtu_MPI produces *.pvtu files and *.vtu files. The user needs to specify list of scalars and vectors and subname.
*/

struct PVD_output {
    char * subname;
    double myt;
    scalar * list;
    vector * vlist;
    vector * fvlist;
};

struct stat st = {0};
static int iter_fp=0;
static float file_timesteps[9999];
void output_vtu_MPI(struct PVD_output o){
    scalar * list = o.list;
    vector * vlist = o.vlist;
    vector * fvlist = o.fvlist;
    char * subname = o.subname;
    double myt = fabs(o.myt);
    int nf = iter_fp;
    char name_vtu[1000];
    if (iter_fp == 0) {
        if (stat("res", &st) == -1) {
            mkdir("res", 0755);
        }
    }
    FILE *fp;
    if (nf>9999) { fprintf(stderr, "too many files, more than 9999"); exit(1); }
    sprintf(name_vtu, "res/%s_%4.4d_n%3.3d.vtu", subname, nf, pid());
    fp = fopen(name_vtu, "w");
    output_vtu_bin_foreach(list, vlist, fvlist, 64, fp, true);//64 and true is useless. It needs to support the interface
    fclose(fp);
    if (pid() == 0) {
        //pvtu file
        char name_pvtu[1000], tmp[1000];
	    sprintf(name_pvtu, "res/%s_0_%4.4d.pvtu", subname, nf);
        sprintf(tmp, "%s_%4.4d", subname, nf);
        fprintf(ferr, "vtk: iter_fp = %d myt = %g\n"
                      "<DataSet timestep=\"%g\" part=\"0\" file=\"%s\"/>\n", nf, myt, myt, name_pvtu);
        fp = fopen(name_pvtu, "w");
        output_pvtu_bin(list, vlist, fvlist, 64, fp, tmp);
        fclose(fp);
        //pvd file with timesteps
        char name_pvd[1000];
        sprintf(name_pvd, "%s.pvd", subname);
        fp = fopen(name_pvd, "w");
        file_timesteps[nf] = myt;
        output_pvd_file(fp, nf, file_timesteps, subname);
        fclose(fp);
    }
    @if _MPI
        MPI_Barrier(MPI_COMM_WORLD);
    @endif
#ifdef DEBUG_OUTPUT_VTU_MPI
    fprintf (ferr, "iter_fp: %d t=%g dt=%g\n", nf, t, dt);
#endif
    iter_fp++;
}


void face_vector2vector(face vector fv, vector mapped_data_lower, vector mapped_data_upper){
//    face vector kappa;
//    kappa = some_face_data;
//    vector mapped_data_lower, mapped_data_upper;
    foreach()
    foreach_dimension(){
        mapped_data_lower.x[] = fv.x[];
        mapped_data_upper.x[] = fv.x[1];
    }
}
