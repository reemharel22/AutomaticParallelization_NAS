//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB BT code. This OpenMP  //
//  C version is developed by the Center for Manycore Programming at Seoul //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB3.3-OMP" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this OpenMP C version to              //
//  cmp@aces.snu.ac.kr                                                     //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//
//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//
//---------------------------------------------------------------------
// program BT
//---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>



#include "header.h"
#include "timers.h"
#include "print_results.h"
/* common /global/ */
double elapsed_time;
int grid_points[3];
logical timeron;
/* common /constants/ */
double tx1;
double tx2;
double tx3;
double ty1;
double ty2;
double ty3;
double tz1;
double tz2;
double tz3;
double dx1;
double dx2;
double dx3;
double dx4;
double dx5;
double dy1;
double dy2;
double dy3;
double dy4;
double dy5;
double dz1;
double dz2;
double dz3;
double dz4;
double dz5;
double dssp;
double dt;
double ce[5][13];
double dxmax;
double dymax;
double dzmax;
double xxcon1;
double xxcon2;
double xxcon3;
double xxcon4;
double xxcon5;
double dx1tx1;
double dx2tx1;
double dx3tx1;
double dx4tx1;
double dx5tx1;
double yycon1;
double yycon2;
double yycon3;
double yycon4;
double yycon5;
double dy1ty1;
double dy2ty1;
double dy3ty1;
double dy4ty1;
double dy5ty1;
double zzcon1;
double zzcon2;
double zzcon3;
double zzcon4;
double zzcon5;
double dz1tz1;
double dz2tz1;
double dz3tz1;
double dz4tz1;
double dz5tz1;
double dnxm1;
double dnym1;
double dnzm1;
double c1c2;
double c1c5;
double c3c4;
double c1345;
double conz1;
double c1;
double c2;
double c3;
double c4;
double c5;
double c4dssp;
double c5dssp;
double dtdssp;
double dttx1;
double dttx2;
double dtty1;
double dtty2;
double dttz1;
double dttz2;
double c2dttx1;
double c2dtty1;
double c2dttz1;
double comz1;
double comz4;
double comz5;
double comz6;
double c3c4tx3;
double c3c4ty3;
double c3c4tz3;
double c2iv;
double con43;
double con16;
// to improve cache performance, grid dimensions padded by 1 
// for even number sizes only.
/* common /fields/ */
double us[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1];
double vs[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1];
double ws[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1];
double qs[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1];
double rho_i[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1];
double square[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1];
double forcing[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1][5];
double u[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1][5];
double rhs[162][162 / 2 * 2 + 1][162 / 2 * 2 + 1][5];
/* common /work_1d/ */
double cuf[162 + 1];
double q[162 + 1];
double ue[162 + 1][5];
double buf[162 + 1][5];
/* common /work_lhs/ */
double fjac[162 + 1][5][5];
double njac[162 + 1][5][5];
double lhs[162 + 1][3][5][5];
double tmp1;
double tmp2;
double tmp3;
//double cuf[PROBLEM_SIZE+1];

int main(int argc,char *argv[])
{
  int i;
  int niter;
  int step;
  double navg;
  double mflops;
  double n3;
  double tmax;
  double t;
  double trecs[11 + 1];
  logical verified;
  char Class;
  char *t_names[11 + 1];
//---------------------------------------------------------------------
// Root node reads input file (if it exists) else takes
// defaults from parameters
//---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag","r")) != ((void *)0)) {
    timeron = true;
    t_names[1] = "total";
    t_names[2] = "rhsx";
    t_names[3] = "rhsy";
    t_names[4] = "rhsz";
    t_names[5] = "rhs";
    t_names[6] = "xsolve";
    t_names[7] = "ysolve";
    t_names[8] = "zsolve";
    t_names[9] = "redist1";
    t_names[10] = "redist2";
    t_names[11] = "add";
    fclose(fp);
  }
   else {
    timeron = false;
  }
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OMP-C) - BT Benchmark\n\n");
  if ((fp = fopen("inputbt.data","r")) != ((void *)0)) {
    int result;
    printf(" Reading from input file inputbt.data\n");
    result = fscanf(fp,"%d",&niter);
    while(fgetc(fp) != '\n')
      ;
    result = fscanf(fp,"%lf",&dt);
    while(fgetc(fp) != '\n')
      ;
    result = fscanf(fp,"%d%d%d\n",&grid_points[0],&grid_points[1],&grid_points[2]);
    fclose(fp);
  }
   else {
    printf(" No input file inputbt.data. Using compiled defaults\n");
    niter = 200;
    dt = 0.0001;
    grid_points[0] = 162;
    grid_points[1] = 162;
    grid_points[2] = 162;
  }
  printf(" Size: %4dx%4dx%4d\n",grid_points[0],grid_points[1],grid_points[2]);
  printf(" Iterations: %4d       dt: %11.7f\n",niter,dt);
  printf(" Number of available threads: %5d\n",(omp_get_max_threads()));
  printf("\n");
  if (grid_points[0] > 162 || grid_points[1] > 162 || grid_points[2] > 162) {
    printf(" %d, %d, %d\n",grid_points[0],grid_points[1],grid_points[2]);
    printf(" Problem size too big for compiled array sizes\n");
    return 0;
  }
  set_constants();
  for (i = 1; i <= 11; i += 1) {
    timer_clear(i);
  }
  initialize();
  exact_rhs();
//---------------------------------------------------------------------
// do one time step to touch all code, and reinitialize
//---------------------------------------------------------------------
  adi();
  initialize();
  for (i = 1; i <= 11; i += 1) {
    timer_clear(i);
  }
  timer_start(1);
  for (step = 1; step <= niter; step += 1) {
    if (step % 20 == 0 || step == 1) {
      printf(" Time step %4d\n",step);
    }
    adi();
  }
  timer_stop(1);
  tmax = timer_read(1);
  verify(niter,&Class,&verified);
  n3 = 1.0 * grid_points[0] * grid_points[1] * grid_points[2];
  navg = (grid_points[0] + grid_points[1] + grid_points[2]) / 3.0;
  if (tmax != 0.0) {
    mflops = 1.0e-6 * ((double )niter) * (3478.8 * n3 - 17655.7 * (navg * navg) + 28023.7 * navg) / tmax;
  }
   else {
    mflops = 0.0;
  }
  print_results("BT",Class,grid_points[0],grid_points[1],grid_points[2],niter,tmax,mflops,"          floating point",verified,"3.3.1","23 Oct 2018","icc","$(CC)","-lm","-I../common","-g -Wall -O3 -qopenmp -mcmodel=medium","-O3 -qopenmp -mcmodel=medium","(none)");
//---------------------------------------------------------------------
// More timers
//---------------------------------------------------------------------
  if (timeron) {
    for (i = 1; i <= 11; i += 1) {
      trecs[i] = timer_read(i);
    }
    if (tmax == 0.0) 
      tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = 1; i <= 11; i += 1) {
      printf("  %-8s:%9.3f  (%6.2f%%)\n",t_names[i],trecs[i],trecs[i] * 100. / tmax);
      if (i == 5) {
        t = trecs[2] + trecs[3] + trecs[4];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n","sub-rhs",t,t * 100. / tmax);
        t = trecs[5] - t;
        printf("    --> %8s:%9.3f  (%6.2f%%)\n","rest-rhs",t,t * 100. / tmax);
      }
       else if (i == 8) {
        t = trecs[8] - trecs[9] - trecs[10];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n","sub-zsol",t,t * 100. / tmax);
      }
       else if (i == 10) {
        t = trecs[9] + trecs[10];
        printf("    --> %8s:%9.3f  (%6.2f%%)\n","redist",t,t * 100. / tmax);
      }
    }
  }
  return 0;
}
