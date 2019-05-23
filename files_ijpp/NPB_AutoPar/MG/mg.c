//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB MG code. This OpenMP  //
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
// program mg
//---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#include "globals.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#include "omp.h" 
static void setup(int *n1,int *n2,int *n3);
static void mg3P(double u[],double v[],double r[],double a[4],double c[4],int n1,int n2,int n3);
static void psinv(void *or,void *ou,int n1,int n2,int n3,double c[4],int k);
static void resid(void *ou,void *ov,void *or,int n1,int n2,int n3,double a[4],int k);
static void rprj3(void *or,int m1k,int m2k,int m3k,void *os,int m1j,int m2j,int m3j,int k);
static void interp(void *oz,int mm1,int mm2,int mm3,void *ou,int n1,int n2,int n3,int k);
static void norm2u3(void *or,int n1,int n2,int n3,double *rnm2,double *rnmu,int nx,int ny,int nz);
static void rep_nrm(void *u,int n1,int n2,int n3,char *title,int kk);
static void comm3(void *ou,int n1,int n2,int n3,int kk);
static void zran3(void *oz,int n1,int n2,int n3,int nx1,int ny1,int k);
static void showall(void *oz,int n1,int n2,int n3);
static double power(double a,int n);
static void bubble(double ten[][2],int j1[][2],int j2[][2],int j3[][2],int m,int ind);
static void zero3(void *oz,int n1,int n2,int n3);
//-------------------------------------------------------------------------c
// These arrays are in common because they are quite large
// and probably shouldn't be allocated on the stack. They
// are always passed as subroutine args. 
//-------------------------------------------------------------------------c
/* commcon /noautom/ */
static double u[(1 * (2 + (1 << 9)) * (2 + (1 << 9)) * (2 + (1 << 9)) + (2 + (1 << 9)) * (2 + (1 << 9)) + 5 * (2 + (1 << 9)) + 7 * 9 + 6) / 7 * 8];
static double v[(1 * (2 + (1 << 9)) * (2 + (1 << 9)) * (2 + (1 << 9)) + (2 + (1 << 9)) * (2 + (1 << 9)) + 5 * (2 + (1 << 9)) + 7 * 9 + 6) / 7 * 8];
static double r[(1 * (2 + (1 << 9)) * (2 + (1 << 9)) * (2 + (1 << 9)) + (2 + (1 << 9)) * (2 + (1 << 9)) + 5 * (2 + (1 << 9)) + 7 * 9 + 6) / 7 * 8];
/* common /grid/ */
static int is1;
static int is2;
static int is3;
static int ie1;
static int ie2;
static int ie3;
/* common /rans_save/ starts */
double starts[2 + (1 << 9)];

int main()
{
//-------------------------------------------------------------------------c
// k is the current level. It is passed down through subroutine args
// and is NOT global. it is the current iteration
//-------------------------------------------------------------------------c
  int k;
  int it;
  double t;
  double tinit;
  double mflops;
  double a[4];
  double c[4];
  double rnm2;
  double rnmu;
  double old2;
  double oldu;
  double epsilon;
  int n1;
  int n2;
  int n3;
  int nit;
  double nn;
  double verify_value;
  double err;
  logical verified;
  int i;
  char *t_names[10];
  double tmax;
  for (i = 0; i <= 9; i += 1) {
    timer_clear(i);
  }
  timer_start(0);
//---------------------------------------------------------------------
// Read in and broadcast input data
//---------------------------------------------------------------------
  FILE *fp;
  if ((fp = fopen("timer.flag","r")) != ((void *)0)) {
    timeron = true;
    t_names[0] = "init";
    t_names[1] = "benchmk";
    t_names[2] = "mg3P";
    t_names[3] = "psinv";
    t_names[4] = "resid";
    t_names[6] = "rprj3";
    t_names[7] = "interp";
    t_names[8] = "norm2";
    t_names[9] = "comm3";
    fclose(fp);
  }
   else {
    timeron = false;
  }
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OMP-C) - MG Benchmark\n\n");
  if ((fp = fopen("mg.input","r")) != ((void *)0)) {
    int result;
    printf(" Reading from input file mg.input\n");
    result = fscanf(fp,"%d\n",&lt);
    while(fgetc(fp) != '\n')
      ;
    result = fscanf(fp,"%d%d%d",&nx[lt],&ny[lt],&nz[lt]);
    while(fgetc(fp) != '\n')
      ;
    result = fscanf(fp,"%d",&nit);
    while(fgetc(fp) != '\n')
      ;
    for (i = 0; i <= 7; i += 1) {
      result = fscanf(fp,"%d",&debug_vec[i]);
    }
    fclose(fp);
  }
   else {
    printf(" No input file. Using compiled defaults \n");
    lt = 9;
    nit = 20;
    nx[lt] = 512;
    ny[lt] = 512;
    nz[lt] = 512;
    
#pragma omp parallel for private (i)
    for (i = 0; i <= 7; i += 1) {
      debug_vec[i] = 0;
    }
  }
  if (nx[lt] != ny[lt] || nx[lt] != nz[lt]) {
    Class = 'U';
  }
   else if (nx[lt] == 32 && nit == 4) {
    Class = 'S';
  }
   else if (nx[lt] == 128 && nit == 4) {
    Class = 'W';
  }
   else if (nx[lt] == 256 && nit == 4) {
    Class = 'A';
  }
   else if (nx[lt] == 256 && nit == 20) {
    Class = 'B';
  }
   else if (nx[lt] == 512 && nit == 20) {
    Class = 'C';
  }
   else if (nx[lt] == 1024 && nit == 50) {
    Class = 'D';
  }
   else if (nx[lt] == 2048 && nit == 50) {
    Class = 'E';
  }
   else {
    Class = 'U';
  }
//---------------------------------------------------------------------
// Use these for debug info:
//---------------------------------------------------------------------
//    debug_vec(0) = 1 !=> report all norms
//    debug_vec(1) = 1 !=> some setup information
//    debug_vec(1) = 2 !=> more setup information
//    debug_vec(2) = k => at level k or below, show result of resid
//    debug_vec(3) = k => at level k or below, show result of psinv
//    debug_vec(4) = k => at level k or below, show result of rprj
//    debug_vec(5) = k => at level k or below, show result of interp
//    debug_vec(6) = 1 => (unused)
//    debug_vec(7) = 1 => (unused)
//---------------------------------------------------------------------
  a[0] = - 8.0 / 3.0;
  a[1] = 0.0;
  a[2] = 1.0 / 6.0;
  a[3] = 1.0 / 12.0;
  if (Class == 'A' || Class == 'S' || Class == 'W') {
//---------------------------------------------------------------------
// Coefficients for the S(a) smoother
//---------------------------------------------------------------------
    c[0] = - 3.0 / 8.0;
    c[1] = +1.0 / 32.0;
    c[2] = - 1.0 / 64.0;
    c[3] = 0.0;
  }
   else {
//---------------------------------------------------------------------
// Coefficients for the S(b) smoother
//---------------------------------------------------------------------
    c[0] = - 3.0 / 17.0;
    c[1] = +1.0 / 33.0;
    c[2] = - 1.0 / 61.0;
    c[3] = 0.0;
  }
  lb = 1;
  k = lt;
  setup(&n1,&n2,&n3);
  zero3(u,n1,n2,n3);
  zran3(v,n1,n2,n3,nx[lt],ny[lt],k);
  norm2u3(v,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);
//  printf("\n");
//  printf(" norms of random v are\n");
//  printf("%4d%19.2f%19.2e\n", 0, rnm2, rnmu);
//  printf(" about to evaluate resid, k=%d\n", k);
  printf(" Size: %4dx%4dx%4d  (class %c)\n",nx[lt],ny[lt],nz[lt],Class);
  printf(" Iterations:                  %5d\n",nit);
  printf(" Number of available threads: %5d\n",(omp_get_max_threads()));
  printf("\n");
  resid(u,v,r,n1,n2,n3,a,k);
  norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);
  old2 = rnm2;
  oldu = rnmu;
//---------------------------------------------------------------------
// One iteration for startup
//---------------------------------------------------------------------
  mg3P(u,v,r,a,c,n1,n2,n3);
  resid(u,v,r,n1,n2,n3,a,k);
  setup(&n1,&n2,&n3);
  zero3(u,n1,n2,n3);
  zran3(v,n1,n2,n3,nx[lt],ny[lt],k);
  timer_stop(0);
  tinit = timer_read(0);
  printf(" Initialization time: %15.3f seconds\n\n",tinit);
  for (i = 1; i <= 9; i += 1) {
    timer_clear(i);
  }
  timer_start(1);
  if (timeron) 
    timer_start(5);
  resid(u,v,r,n1,n2,n3,a,k);
  if (timeron) 
    timer_stop(5);
  norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);
  old2 = rnm2;
  oldu = rnmu;
  for (it = 1; it <= nit; it += 1) {
    if (it == 1 || it == nit || it % 5 == 0) {
      printf("  iter %3d\n",it);
    }
    if (timeron) 
      timer_start(2);
    mg3P(u,v,r,a,c,n1,n2,n3);
    if (timeron) 
      timer_stop(2);
    if (timeron) 
      timer_start(5);
    resid(u,v,r,n1,n2,n3,a,k);
    if (timeron) 
      timer_stop(5);
  }
  norm2u3(r,n1,n2,n3,&rnm2,&rnmu,nx[lt],ny[lt],nz[lt]);
  timer_stop(1);
  t = timer_read(1);
  verified = false;
  verify_value = 0.0;
  printf("\n Benchmark completed\n");
  epsilon = 1.0e-8;
  if (Class != 'U') {
    if (Class == 'S') {
      verify_value = 0.5307707005734e-04;
    }
     else if (Class == 'W') {
      verify_value = 0.6467329375339e-05;
    }
     else if (Class == 'A') {
      verify_value = 0.2433365309069e-05;
    }
     else if (Class == 'B') {
      verify_value = 0.1800564401355e-05;
    }
     else if (Class == 'C') {
      verify_value = 0.5706732285740e-06;
    }
     else if (Class == 'D') {
      verify_value = 0.1583275060440e-09;
    }
     else if (Class == 'E') {
      verify_value = 0.5630442584711e-10;
    }
    err = fabs(rnm2 - verify_value) / verify_value;
    if (err <= epsilon) {
      verified = true;
      printf(" VERIFICATION SUCCESSFUL\n");
      printf(" L2 Norm is %20.13E\n",rnm2);
      printf(" Error is   %20.13E\n",err);
    }
     else {
      verified = false;
      printf(" VERIFICATION FAILED\n");
      printf(" L2 Norm is             %20.13E\n",rnm2);
      printf(" The correct L2 Norm is %20.13E\n",verify_value);
    }
  }
   else {
    verified = false;
    printf(" Problem size unknown\n");
    printf(" NO VERIFICATION PERFORMED\n");
    printf(" L2 Norm is %20.13E\n",rnm2);
  }
  nn = 1.0 * nx[lt] * ny[lt] * nz[lt];
  if (t != 0.0) {
    mflops = 58.0 * nit * nn * 1.0e-6 / t;
  }
   else {
    mflops = 0.0;
  }
  print_results("MG",Class,nx[lt],ny[lt],nz[lt],nit,t,mflops,"          floating point",verified,"3.3.1","23 Oct 2018","icc","$(CC)","-lm","-I../common","-g -Wall -O3 -qopenmp -mcmodel=medium","-O3 -qopenmp -mcmodel=medium","randdp");
//---------------------------------------------------------------------
// More timers
//---------------------------------------------------------------------
  if (timeron) {
    tmax = timer_read(1);
    if (tmax == 0.0) 
      tmax = 1.0;
    printf("  SECTION   Time (secs)\n");
    for (i = 1; i <= 9; i += 1) {
      t = timer_read(i);
      if (i == 5) {
        t = timer_read(4) - t;
        printf("    --> %8s:%9.3f  (%6.2f%%)\n","mg-resid",t,t * 100. / tmax);
      }
       else {
        printf("  %-8s:%9.3f  (%6.2f%%)\n",t_names[i],t,t * 100. / tmax);
      }
    }
  }
  return 0;
}

static void setup(int *n1,int *n2,int *n3)
{
  int k;
  int j;
  int ax;
  int mi[10 + 1][3];
  int ng[10 + 1][3];
  ng[lt][0] = nx[lt];
  ng[lt][1] = ny[lt];
  ng[lt][2] = nz[lt];
  for (k = lt - 1; k >= 1; k += -1) {
    
#pragma omp parallel for private (ax)
    for (ax = 0; ax <= 2; ax += 1) {
      ng[k][ax] = ng[k + 1][ax] / 2;
    }
  }
  
#pragma omp parallel for private (k)
  for (k = lt; k >= 1; k += -1) {
    nx[k] = ng[k][0];
    ny[k] = ng[k][1];
    nz[k] = ng[k][2];
  }
  
#pragma omp parallel for private (k,ax)
  for (k = lt; k >= 1; k += -1) {
    
#pragma omp parallel for private (ax)
    for (ax = 0; ax <= 2; ax += 1) {
      mi[k][ax] = 2 + ng[k][ax];
    }
    m1[k] = mi[k][0];
    m2[k] = mi[k][1];
    m3[k] = mi[k][2];
  }
  k = lt;
  is1 = 2 + ng[k][0] - ng[lt][0];
  ie1 = 1 + ng[k][0];
   *n1 = 3 + ie1 - is1;
  is2 = 2 + ng[k][1] - ng[lt][1];
  ie2 = 1 + ng[k][1];
   *n2 = 3 + ie2 - is2;
  is3 = 2 + ng[k][2] - ng[lt][2];
  ie3 = 1 + ng[k][2];
   *n3 = 3 + ie3 - is3;
  ir[lt] = 0;
  for (j = lt - 1; j >= 1; j += -1) {
    ir[j] = ir[j + 1] + 1 * m1[j + 1] * m2[j + 1] * m3[j + 1];
  }
  if (debug_vec[1] >= 1) {
    printf(" in setup, \n");
    printf(" k  lt  nx  ny  nz  n1  n2  n3 is1 is2 is3 ie1 ie2 ie3\n");
    printf("%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d%4d\n",k,lt,ng[k][0],ng[k][1],ng[k][2], *n1, *n2, *n3,is1,is2,is3,ie1,ie2,ie3);
  }
}
//---------------------------------------------------------------------
// multigrid V-cycle routine
//---------------------------------------------------------------------

static void mg3P(double u[],double v[],double r[],double a[4],double c[4],int n1,int n2,int n3)
{
  int j;
  int k;
//---------------------------------------------------------------------
// down cycle.
// restrict the residual from the find grid to the coarse
//---------------------------------------------------------------------
  for (k = lt; k >= lb + 1; k += -1) {
    j = k - 1;
    rprj3((&r[ir[k]]),m1[k],m2[k],m3[k],(&r[ir[j]]),m1[j],m2[j],m3[j],k);
  }
  k = lb;
//---------------------------------------------------------------------
// compute an approximate solution on the coarsest grid
//---------------------------------------------------------------------
  zero3((&u[ir[k]]),m1[k],m2[k],m3[k]);
  psinv((&r[ir[k]]),(&u[ir[k]]),m1[k],m2[k],m3[k],c,k);
  for (k = lb + 1; k <= lt - 1; k += 1) {
    j = k - 1;
//---------------------------------------------------------------------
// prolongate from level k-1  to k
//---------------------------------------------------------------------
    zero3((&u[ir[k]]),m1[k],m2[k],m3[k]);
    interp((&u[ir[j]]),m1[j],m2[j],m3[j],(&u[ir[k]]),m1[k],m2[k],m3[k],k);
//---------------------------------------------------------------------
// compute residual for level k
//---------------------------------------------------------------------
    resid((&u[ir[k]]),(&r[ir[k]]),(&r[ir[k]]),m1[k],m2[k],m3[k],a,k);
//---------------------------------------------------------------------
// apply smoother
//---------------------------------------------------------------------
    psinv((&r[ir[k]]),(&u[ir[k]]),m1[k],m2[k],m3[k],c,k);
  }
  j = lt - 1;
  k = lt;
  interp((&u[ir[j]]),m1[j],m2[j],m3[j],u,n1,n2,n3,k);
  resid(u,v,r,n1,n2,n3,a,k);
  psinv(r,u,n1,n2,n3,c,k);
}
//---------------------------------------------------------------------
// psinv applies an approximate inverse as smoother:  u = u + Cr
//
// This  implementation costs  15A + 4M per result, where
// A and M denote the costs of Addition and Multiplication.  
// Presuming coefficient c(3) is zero (the NPB assumes this,
// but it is thus not a general case), 2A + 1M may be eliminated,
// resulting in 13A + 3M.
// Note that this vectorizes, and is also fine for cache 
// based machines.  
//---------------------------------------------------------------------

static void psinv(void *or,void *ou,int n1,int n2,int n3,double c[4],int k)
{
  double (*r)[n2][n1] = ((double (*)[n2][n1])or);
  double (*u)[n2][n1] = ((double (*)[n2][n1])ou);
  int i3;
  int i2;
  int i1;
  double r1[514 + 1];
  double r2[514 + 1];
  if (timeron) 
    timer_start(3);
  for (i3 = 1; i3 <= n3 - 1 - 1; i3 += 1) {
    for (i2 = 1; i2 <= n2 - 1 - 1; i2 += 1) {
      
#pragma omp parallel for private (i1)
      for (i1 = 0; i1 <= n1 - 1; i1 += 1) {
        r1[i1] = r[i3][i2 - 1][i1] + r[i3][i2 + 1][i1] + r[i3 - 1][i2][i1] + r[i3 + 1][i2][i1];
        r2[i1] = r[i3 - 1][i2 - 1][i1] + r[i3 - 1][i2 + 1][i1] + r[i3 + 1][i2 - 1][i1] + r[i3 + 1][i2 + 1][i1];
      }
      
#pragma omp parallel for private (i1)
      for (i1 = 1; i1 <= n1 - 1 - 1; i1 += 1) {
        u[i3][i2][i1] = u[i3][i2][i1] + c[0] * r[i3][i2][i1] + c[1] * (r[i3][i2][i1 - 1] + r[i3][i2][i1 + 1] + r1[i1]) + c[2] * (r2[i1] + r1[i1 - 1] + r1[i1 + 1]);
//--------------------------------------------------------------------
// Assume c[3] = 0    (Enable line below if c[3] not= 0)
//--------------------------------------------------------------------
//            + c[3] * ( r2[i1-1] + r2[i1+1] )
//--------------------------------------------------------------------
      }
    }
  }
  if (timeron) 
    timer_stop(3);
//---------------------------------------------------------------------
// exchange boundary points
//---------------------------------------------------------------------
  comm3(u,n1,n2,n3,k);
  if (debug_vec[0] >= 1) {
    rep_nrm(u,n1,n2,n3,"   psinv",k);
  }
  if (debug_vec[3] >= k) {
    showall(u,n1,n2,n3);
  }
}
//---------------------------------------------------------------------
// resid computes the residual:  r = v - Au
//
// This  implementation costs  15A + 4M per result, where
// A and M denote the costs of Addition (or Subtraction) and 
// Multiplication, respectively. 
// Presuming coefficient a(1) is zero (the NPB assumes this,
// but it is thus not a general case), 3A + 1M may be eliminated,
// resulting in 12A + 3M.
// Note that this vectorizes, and is also fine for cache 
// based machines.  
//---------------------------------------------------------------------

static void resid(void *ou,void *ov,void *or,int n1,int n2,int n3,double a[4],int k)
{
  double (*u)[n2][n1] = ((double (*)[n2][n1])ou);
  double (*v)[n2][n1] = ((double (*)[n2][n1])ov);
  double (*r)[n2][n1] = ((double (*)[n2][n1])or);
  int i3;
  int i2;
  int i1;
  double u1[514 + 1];
  double u2[514 + 1];
  if (timeron) 
    timer_start(4);
  for (i3 = 1; i3 <= n3 - 1 - 1; i3 += 1) {
    for (i2 = 1; i2 <= n2 - 1 - 1; i2 += 1) {
      
#pragma omp parallel for private (i1)
      for (i1 = 0; i1 <= n1 - 1; i1 += 1) {
        u1[i1] = u[i3][i2 - 1][i1] + u[i3][i2 + 1][i1] + u[i3 - 1][i2][i1] + u[i3 + 1][i2][i1];
        u2[i1] = u[i3 - 1][i2 - 1][i1] + u[i3 - 1][i2 + 1][i1] + u[i3 + 1][i2 - 1][i1] + u[i3 + 1][i2 + 1][i1];
      }
      
#pragma omp parallel for private (i1)
      for (i1 = 1; i1 <= n1 - 1 - 1; i1 += 1) {
        r[i3][i2][i1] = v[i3][i2][i1] - a[0] * u[i3][i2][i1] - a[2] * (u2[i1] + u1[i1 - 1] + u1[i1 + 1]) - a[3] * (u2[i1 - 1] + u2[i1 + 1]);
//-------------------------------------------------------------------
//  Assume a[1] = 0      (Enable 2 lines below if a[1] not= 0)
//-------------------------------------------------------------------
//            - a[1] * ( u[i3][i2][i1-1] + u[i3][i2][i1+1]
//                     + u1[i1] )
//-------------------------------------------------------------------
      }
    }
  }
  if (timeron) 
    timer_stop(4);
//---------------------------------------------------------------------
// exchange boundary data
//---------------------------------------------------------------------
  comm3(r,n1,n2,n3,k);
  if (debug_vec[0] >= 1) {
    rep_nrm(r,n1,n2,n3,"   resid",k);
  }
  if (debug_vec[2] >= k) {
    showall(r,n1,n2,n3);
  }
}
//---------------------------------------------------------------------
// rprj3 projects onto the next coarser grid, 
// using a trilinear Finite Element projection:  s = r' = P r
//     
// This  implementation costs  20A + 4M per result, where
// A and M denote the costs of Addition and Multiplication.  
// Note that this vectorizes, and is also fine for cache 
// based machines.  
//---------------------------------------------------------------------

static void rprj3(void *or,int m1k,int m2k,int m3k,void *os,int m1j,int m2j,int m3j,int k)
{
  double (*r)[m2k][m1k] = ((double (*)[m2k][m1k])or);
  double (*s)[m2j][m1j] = ((double (*)[m2j][m1j])os);
  int j3;
  int j2;
  int j1;
  int i3;
  int i2;
  int i1;
  int d1;
  int d2;
  int d3;
  int j;
  double x1[514 + 1];
  double y1[514 + 1];
  double x2;
  double y2;
  if (timeron) 
    timer_start(6);
  if (m1k == 3) {
    d1 = 2;
  }
   else {
    d1 = 1;
  }
  if (m2k == 3) {
    d2 = 2;
  }
   else {
    d2 = 1;
  }
  if (m3k == 3) {
    d3 = 2;
  }
   else {
    d3 = 1;
  }
  for (j3 = 1; j3 <= m3j - 1 - 1; j3 += 1) {
    i3 = 2 * j3 - d3;
    for (j2 = 1; j2 <= m2j - 1 - 1; j2 += 1) {
      i2 = 2 * j2 - d2;
      for (j1 = 1; j1 <= m1j - 1; j1 += 1) {
        i1 = 2 * j1 - d1;
        x1[i1] = r[i3 + 1][i2][i1] + r[i3 + 1][i2 + 2][i1] + r[i3][i2 + 1][i1] + r[i3 + 2][i2 + 1][i1];
        y1[i1] = r[i3][i2][i1] + r[i3 + 2][i2][i1] + r[i3][i2 + 2][i1] + r[i3 + 2][i2 + 2][i1];
      }
      
#pragma omp parallel for private (i1,x2,y2,j1) firstprivate (i3,i2,d1)
      for (j1 = 1; j1 <= m1j - 1 - 1; j1 += 1) {
        i1 = 2 * j1 - d1;
        y2 = r[i3][i2][i1 + 1] + r[i3 + 2][i2][i1 + 1] + r[i3][i2 + 2][i1 + 1] + r[i3 + 2][i2 + 2][i1 + 1];
        x2 = r[i3 + 1][i2][i1 + 1] + r[i3 + 1][i2 + 2][i1 + 1] + r[i3][i2 + 1][i1 + 1] + r[i3 + 2][i2 + 1][i1 + 1];
        s[j3][j2][j1] = 0.5 * r[i3 + 1][i2 + 1][i1 + 1] + 0.25 * (r[i3 + 1][i2 + 1][i1] + r[i3 + 1][i2 + 1][i1 + 2] + x2) + 0.125 * (x1[i1] + x1[i1 + 2] + y2) + 0.0625 * (y1[i1] + y1[i1 + 2]);
      }
    }
  }
  if (timeron) 
    timer_stop(6);
  j = k - 1;
  comm3(s,m1j,m2j,m3j,j);
  if (debug_vec[0] >= 1) {
    rep_nrm(s,m1j,m2j,m3j,"   rprj3",k - 1);
  }
  if (debug_vec[4] >= k) {
    showall(s,m1j,m2j,m3j);
  }
}
//---------------------------------------------------------------------
// interp adds the trilinear interpolation of the correction
// from the coarser grid to the current approximation:  u = u + Qu'
//     
// Observe that this  implementation costs  16A + 4M, where
// A and M denote the costs of Addition and Multiplication.  
// Note that this vectorizes, and is also fine for cache 
// based machines.  Vector machines may get slightly better 
// performance however, with 8 separate "do i1" loops, rather than 4.
//---------------------------------------------------------------------

static void interp(void *oz,int mm1,int mm2,int mm3,void *ou,int n1,int n2,int n3,int k)
{
  double (*z)[mm2][mm1] = ((double (*)[mm2][mm1])oz);
  double (*u)[n2][n1] = ((double (*)[n2][n1])ou);
  int i3;
  int i2;
  int i1;
  int d1;
  int d2;
  int d3;
  int t1;
  int t2;
  int t3;
// note that m = 1037 in globals.h but for this only need to be
// 535 to handle up to 1024^3
//      integer m
//      parameter( m=535 )
  double z1[514 + 1];
  double z2[514 + 1];
  double z3[514 + 1];
  if (timeron) 
    timer_start(7);
  if (n1 != 3 && n2 != 3 && n3 != 3) {
    for (i3 = 0; i3 <= mm3 - 1 - 1; i3 += 1) {
      for (i2 = 0; i2 <= mm2 - 1 - 1; i2 += 1) {
        
#pragma omp parallel for private (i1)
        for (i1 = 0; i1 <= mm1 - 1; i1 += 1) {
          z1[i1] = z[i3][i2 + 1][i1] + z[i3][i2][i1];
          z2[i1] = z[i3 + 1][i2][i1] + z[i3][i2][i1];
          z3[i1] = z[i3 + 1][i2 + 1][i1] + z[i3 + 1][i2][i1] + z1[i1];
        }
        
#pragma omp parallel for private (i1)
        for (i1 = 0; i1 <= mm1 - 1 - 1; i1 += 1) {
          u[2 * i3][2 * i2][2 * i1] = u[2 * i3][2 * i2][2 * i1] + z[i3][i2][i1];
          u[2 * i3][2 * i2][2 * i1 + 1] = u[2 * i3][2 * i2][2 * i1 + 1] + 0.5 * (z[i3][i2][i1 + 1] + z[i3][i2][i1]);
        }
        
#pragma omp parallel for private (i1)
        for (i1 = 0; i1 <= mm1 - 1 - 1; i1 += 1) {
          u[2 * i3][2 * i2 + 1][2 * i1] = u[2 * i3][2 * i2 + 1][2 * i1] + 0.5 * z1[i1];
          u[2 * i3][2 * i2 + 1][2 * i1 + 1] = u[2 * i3][2 * i2 + 1][2 * i1 + 1] + 0.25 * (z1[i1] + z1[i1 + 1]);
        }
        
#pragma omp parallel for private (i1)
        for (i1 = 0; i1 <= mm1 - 1 - 1; i1 += 1) {
          u[2 * i3 + 1][2 * i2][2 * i1] = u[2 * i3 + 1][2 * i2][2 * i1] + 0.5 * z2[i1];
          u[2 * i3 + 1][2 * i2][2 * i1 + 1] = u[2 * i3 + 1][2 * i2][2 * i1 + 1] + 0.25 * (z2[i1] + z2[i1 + 1]);
        }
        
#pragma omp parallel for private (i1)
        for (i1 = 0; i1 <= mm1 - 1 - 1; i1 += 1) {
          u[2 * i3 + 1][2 * i2 + 1][2 * i1] = u[2 * i3 + 1][2 * i2 + 1][2 * i1] + 0.25 * z3[i1];
          u[2 * i3 + 1][2 * i2 + 1][2 * i1 + 1] = u[2 * i3 + 1][2 * i2 + 1][2 * i1 + 1] + 0.125 * (z3[i1] + z3[i1 + 1]);
        }
      }
    }
  }
   else {
    if (n1 == 3) {
      d1 = 2;
      t1 = 1;
    }
     else {
      d1 = 1;
      t1 = 0;
    }
    if (n2 == 3) {
      d2 = 2;
      t2 = 1;
    }
     else {
      d2 = 1;
      t2 = 0;
    }
    if (n3 == 3) {
      d3 = 2;
      t3 = 1;
    }
     else {
      d3 = 1;
      t3 = 0;
    }
{
      
#pragma omp parallel for private (i3,i2,i1)
      for (i3 = d3; i3 <= mm3 - 1; i3 += 1) {
        
#pragma omp parallel for private (i2,i1)
        for (i2 = d2; i2 <= mm2 - 1; i2 += 1) {
          
#pragma omp parallel for private (i1)
          for (i1 = d1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - d3 - 1][2 * i2 - d2 - 1][2 * i1 - d1 - 1] = u[2 * i3 - d3 - 1][2 * i2 - d2 - 1][2 * i1 - d1 - 1] + z[i3 - 1][i2 - 1][i1 - 1];
          }
          
#pragma omp parallel for private (i1) firstprivate (d3,t1)
          for (i1 = 1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - d3 - 1][2 * i2 - d2 - 1][2 * i1 - t1 - 1] = u[2 * i3 - d3 - 1][2 * i2 - d2 - 1][2 * i1 - t1 - 1] + 0.5 * (z[i3 - 1][i2 - 1][i1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
        }
        
#pragma omp parallel for private (i2,i1) firstprivate (d1)
        for (i2 = 1; i2 <= mm2 - 1; i2 += 1) {
          
#pragma omp parallel for private (i1)
          for (i1 = d1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - d3 - 1][2 * i2 - t2 - 1][2 * i1 - d1 - 1] = u[2 * i3 - d3 - 1][2 * i2 - t2 - 1][2 * i1 - d1 - 1] + 0.5 * (z[i3 - 1][i2][i1 - 1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
          
#pragma omp parallel for private (i1) firstprivate (d3,t1,t2)
          for (i1 = 1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - d3 - 1][2 * i2 - t2 - 1][2 * i1 - t1 - 1] = u[2 * i3 - d3 - 1][2 * i2 - t2 - 1][2 * i1 - t1 - 1] + 0.25 * (z[i3 - 1][i2][i1] + z[i3 - 1][i2 - 1][i1] + z[i3 - 1][i2][i1 - 1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
        }
      }
      
#pragma omp parallel for private (i3,i2,i1) firstprivate (d2)
      for (i3 = 1; i3 <= mm3 - 1; i3 += 1) {
        
#pragma omp parallel for private (i2,i1)
        for (i2 = d2; i2 <= mm2 - 1; i2 += 1) {
          
#pragma omp parallel for private (i1)
          for (i1 = d1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - t3 - 1][2 * i2 - d2 - 1][2 * i1 - d1 - 1] = u[2 * i3 - t3 - 1][2 * i2 - d2 - 1][2 * i1 - d1 - 1] + 0.5 * (z[i3][i2 - 1][i1 - 1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
          
#pragma omp parallel for private (i1) firstprivate (t1,t3)
          for (i1 = 1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - t3 - 1][2 * i2 - d2 - 1][2 * i1 - t1 - 1] = u[2 * i3 - t3 - 1][2 * i2 - d2 - 1][2 * i1 - t1 - 1] + 0.25 * (z[i3][i2 - 1][i1] + z[i3][i2 - 1][i1 - 1] + z[i3 - 1][i2 - 1][i1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
        }
        
#pragma omp parallel for private (i2,i1) firstprivate (d1)
        for (i2 = 1; i2 <= mm2 - 1; i2 += 1) {
          
#pragma omp parallel for private (i1)
          for (i1 = d1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - t3 - 1][2 * i2 - t2 - 1][2 * i1 - d1 - 1] = u[2 * i3 - t3 - 1][2 * i2 - t2 - 1][2 * i1 - d1 - 1] + 0.25 * (z[i3][i2][i1 - 1] + z[i3][i2 - 1][i1 - 1] + z[i3 - 1][i2][i1 - 1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
          
#pragma omp parallel for private (i1) firstprivate (t1,t2,t3)
          for (i1 = 1; i1 <= mm1 - 1; i1 += 1) {
            u[2 * i3 - t3 - 1][2 * i2 - t2 - 1][2 * i1 - t1 - 1] = u[2 * i3 - t3 - 1][2 * i2 - t2 - 1][2 * i1 - t1 - 1] + 0.125 * (z[i3][i2][i1] + z[i3][i2 - 1][i1] + z[i3][i2][i1 - 1] + z[i3][i2 - 1][i1 - 1] + z[i3 - 1][i2][i1] + z[i3 - 1][i2 - 1][i1] + z[i3 - 1][i2][i1 - 1] + z[i3 - 1][i2 - 1][i1 - 1]);
          }
        }
      }
// end parallel
    }
  }
  if (timeron) 
    timer_stop(7);
  if (debug_vec[0] >= 1) {
    rep_nrm(z,mm1,mm2,mm3,"z: inter",k - 1);
    rep_nrm(u,n1,n2,n3,"u: inter",k);
  }
  if (debug_vec[5] >= k) {
    showall(z,mm1,mm2,mm3);
    showall(u,n1,n2,n3);
  }
}
//---------------------------------------------------------------------
// norm2u3 evaluates approximations to the L2 norm and the
// uniform (or L-infinity or Chebyshev) norm, under the
// assumption that the boundaries are periodic or zero.  Add the
// boundaries in with half weight (quarter weight on the edges
// and eighth weight at the corners) for inhomogeneous boundaries.
//---------------------------------------------------------------------

static void norm2u3(void *or,int n1,int n2,int n3,double *rnm2,double *rnmu,int nx,int ny,int nz)
{
  double (*r)[n2][n1] = ((double (*)[n2][n1])or);
  double s;
  double a;
  int i3;
  int i2;
  int i1;
  double dn;
  double max_rnmu;
  if (timeron) 
    timer_start(8);
  dn = 1.0 * nx * ny * nz;
  s = 0.0;
  max_rnmu = 0.0;
{
    double my_rnmu = 0.0;
    for (i3 = 1; i3 <= n3 - 1 - 1; i3 += 1) {
      for (i2 = 1; i2 <= n2 - 1 - 1; i2 += 1) {
        for (i1 = 1; i1 <= n1 - 1 - 1; i1 += 1) {
          s = s + pow(r[i3][i2][i1],2.0);
          a = fabs(r[i3][i2][i1]);
          my_rnmu = (a > my_rnmu?a : my_rnmu);
        }
      }
    }
    if (my_rnmu > max_rnmu) {
      max_rnmu = (my_rnmu > max_rnmu?my_rnmu : max_rnmu);
    }
// end parallel
  }
   *rnmu = max_rnmu;
   *rnm2 = sqrt(s / dn);
  if (timeron) 
    timer_stop(8);
}
//---------------------------------------------------------------------
// report on norm
//---------------------------------------------------------------------

static void rep_nrm(void *u,int n1,int n2,int n3,char *title,int kk)
{
  double rnm2;
  double rnmu;
  norm2u3(u,n1,n2,n3,&rnm2,&rnmu,nx[kk],ny[kk],nz[kk]);
  printf(" Level%2d in %8s: norms =%21.14E%21.14E\n",kk,title,rnm2,rnmu);
}
//---------------------------------------------------------------------
// comm3 organizes the communication on all borders 
//---------------------------------------------------------------------

static void comm3(void *ou,int n1,int n2,int n3,int kk)
{
  double (*u)[n2][n1] = ((double (*)[n2][n1])ou);
  int i1;
  int i2;
  int i3;
  if (timeron) 
    timer_start(9);
{
    
#pragma omp parallel for private (i1,i2,i3) firstprivate (n3)
    for (i3 = 1; i3 <= n3 - 1 - 1; i3 += 1) {
      
#pragma omp parallel for private (i2)
      for (i2 = 1; i2 <= n2 - 1 - 1; i2 += 1) {
        u[i3][i2][0] = u[i3][i2][n1 - 2];
        u[i3][i2][n1 - 1] = u[i3][i2][1];
      }
//  }
//  for (i3 = 1; i3 < n3-1; i3++) {
      
#pragma omp parallel for private (i1)
      for (i1 = 0; i1 <= n1 - 1; i1 += 1) {
        u[i3][0][i1] = u[i3][n2 - 2][i1];
        u[i3][n2 - 1][i1] = u[i3][1][i1];
      }
    }
    
#pragma omp parallel for private (i1,i2) firstprivate (n1,n2)
    for (i2 = 0; i2 <= n2 - 1; i2 += 1) {
      
#pragma omp parallel for private (i1) firstprivate (n3)
      for (i1 = 0; i1 <= n1 - 1; i1 += 1) {
        u[0][i2][i1] = u[n3 - 2][i2][i1];
        u[n3 - 1][i2][i1] = u[1][i2][i1];
      }
    }
// end parallel
  }
  if (timeron) 
    timer_stop(9);
}
//---------------------------------------------------------------------
// zran3  loads +1 at ten randomly chosen points,
// loads -1 at a different ten random points,
// and zero elsewhere.
//---------------------------------------------------------------------

static void zran3(void *oz,int n1,int n2,int n3,int nx1,int ny1,int k)
{
  double (*z)[n2][n1] = ((double (*)[n2][n1])oz);
  int i0;
  int mm0;
  int mm1;
  int i1;
  int i2;
  int i3;
  int d1;
  int e1;
  int e2;
  int e3;
  double xx;
  double x0;
  double x1;
  double a1;
  double a2;
  double ai;
  const int mm = 10;
  const double a = pow(5.0,13.0);
  const double x = 314159265.0;
  double ten[mm][2];
  double best0;
  double best1;
  int i;
  int j1[mm][2];
  int j2[mm][2];
  int j3[mm][2];
  int jg[4][mm][2];
  double rdummy;
  int myid;
  int num_threads;
  a1 = power(a,nx1);
  a2 = power(a,nx1 * ny1);
  zero3(z,n1,n2,n3);
  i = is1 - 2 + nx1 * (is2 - 2 + ny1 * (is3 - 2));
  ai = power(a,i);
  d1 = ie1 - is1 + 1;
  e1 = ie1 - is1 + 2;
  e2 = ie2 - is2 + 2;
  e3 = ie3 - is3 + 2;
  x0 = x;
  rdummy = randlc(&x0,ai);
//---------------------------------------------------------------------
// save the starting seeds for the following loop
//---------------------------------------------------------------------
  for (i3 = 1; i3 <= e3 - 1; i3 += 1) {
    starts[i3] = x0;
    rdummy = randlc(&x0,a2);
  }
//---------------------------------------------------------------------
// fill array
//---------------------------------------------------------------------
  for (i3 = 1; i3 <= e3 - 1; i3 += 1) {
    x1 = starts[i3];
    for (i2 = 1; i2 <= e2 - 1; i2 += 1) {
      xx = x1;
      vranlc(d1,&xx,a,&z[i3][i2][1]);
      rdummy = randlc(&x1,a1);
    }
  }
//---------------------------------------------------------------------
// comm3(z,n1,n2,n3);
// showall(z,n1,n2,n3);
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// each thread looks for twenty candidates
//---------------------------------------------------------------------
  
#pragma omp parallel for private (i)
  for (i = 0; i <= mm - 1; i += 1) {
    ten[i][1] = 0.0;
    j1[i][1] = 0;
    j2[i][1] = 0;
    j3[i][1] = 0;
    ten[i][0] = 1.0;
    j1[i][0] = 0;
    j2[i][0] = 0;
    j3[i][0] = 0;
  }
  for (i3 = 1; i3 <= n3 - 1 - 1; i3 += 1) {
    double (*zi3)[n1] = z[i3];
    for (i2 = 1; i2 <= n2 - 1 - 1; i2 += 1) {
      for (i1 = 1; i1 <= n1 - 1 - 1; i1 += 1) {
        if (zi3[i2][i1] > ten[0][1]) {
          ten[0][1] = zi3[i2][i1];
          j1[0][1] = i1;
          j2[0][1] = i2;
          j3[0][1] = i3;
          bubble(ten,j1,j2,j3,mm,1);
        }
        if (zi3[i2][i1] < ten[0][0]) {
          ten[0][0] = zi3[i2][i1];
          j1[0][0] = i1;
          j2[0][0] = i2;
          j3[0][0] = i3;
          bubble(ten,j1,j2,j3,mm,0);
        }
      }
    }
  }
//---------------------------------------------------------------------
// Now which of these are globally best?
//---------------------------------------------------------------------
  i1 = mm - 1;
  i0 = mm - 1;
  myid = 0;
  myid = omp_get_thread_num();
  num_threads = omp_get_num_threads();
  
#pragma omp parallel for private (i0,i1,best0,best1,i2,i) firstprivate (myid,num_threads)
  for (i = mm - 1; i >= 0; i += -1) {
// ... ORDERED access is required here for sequential consistency
// ... in case that two values are identical.
// ... Since an "ORDERED" section is only defined in OpenMP 2,
// ... we use a dummy loop to emulate ordered access in OpenMP 1.x.
    best1 = 0.0;
    best0 = 1.0;
    for (i2 = 1; i2 <= num_threads; i2 += 1) {{
        if (ten[i1][1] > best1) {
          best1 = ten[i1][1];
          jg[0][i][1] = myid;
        }
        if (ten[i0][0] < best0) {
          best0 = ten[i0][0];
          jg[0][i][0] = myid;
        }
      }
    }
    if (myid == jg[0][i][1]) {
      jg[1][i][1] = j1[i1][1];
      jg[2][i][1] = j2[i1][1];
      jg[3][i][1] = j3[i1][1];
      i1 = i1 - 1;
    }
    if (myid == jg[0][i][0]) {
      jg[1][i][0] = j1[i0][0];
      jg[2][i][0] = j2[i0][0];
      jg[3][i][0] = j3[i0][0];
      i0 = i0 - 1;
    }
  }
//  mm1 = i1+1;
//  mm0 = i0+1;
  mm1 = 0;
  mm0 = 0;
/*
  int cnt = 0;
  printf("  \n");
  printf("  negative charges at\n");
  for (i = 0; i < mm; i++) {
    printf(" (%3d,%3d,%3d)", jg[1][i][0], jg[2][i][0], jg[3][i][0]);
    if (++cnt % 5 == 0) printf("\n");
  }
  cnt = 0;
  printf("  positive charges at\n");
  for (i = 0; i < mm; i++) {
    printf(" (%3d,%3d,%3d)", jg[1][i][1], jg[2][i][1], jg[3][i][1]);
    if (++cnt % 5 == 0) printf("\n");
  }
  cnt = 0;
  printf("  small random numbers were\n");
  for (i = mm-1; i >= 0; i--) {
    printf(" %15.8E", ten[i][0]);
    if (++cnt % 5 == 0) printf("\n");
  }
  cnt = 0;
  printf("  and they were found on processor number\n");
  for (i = mm-1; i >= 0; i--) {
    printf(" %4d", jg[0][i][0]);
    if (++cnt % 10 == 0) printf("\n");
  }
  cnt = 0;
  printf("  large random numbers were\n");
  for (i = mm-1; i >= 0; i--) {
    printf(" %15.8E", ten[i][1]);
    if (++cnt % 5 == 0) printf("\n");
  }
  cnt = 0;
  printf("  and they were found on processor number\n");
  for (i = mm-1; i >= 0; i--) {
    printf(" %4d", jg[0][i][1]);
    if (++cnt % 10 == 0) printf("\n");
  }
  */
  
#pragma omp parallel for private (i1,i2,i3)
  for (i3 = 0; i3 <= n3 - 1; i3 += 1) {
    
#pragma omp parallel for private (i1,i2)
    for (i2 = 0; i2 <= n2 - 1; i2 += 1) {
      
#pragma omp parallel for private (i1)
      for (i1 = 0; i1 <= n1 - 1; i1 += 1) {
        z[i3][i2][i1] = 0.0;
      }
    }
  }
  for (i = mm - 1; i >= mm0; i += -1) {
    z[jg[3][i][0]][jg[2][i][0]][jg[1][i][0]] = - 1.0;
  }
  for (i = mm - 1; i >= mm1; i += -1) {
    z[jg[3][i][1]][jg[2][i][1]][jg[1][i][1]] = +1.0;
  }
  comm3(z,n1,n2,n3,k);
//---------------------------------------------------------------------
// showall(z,n1,n2,n3);
//---------------------------------------------------------------------
}

static void showall(void *oz,int n1,int n2,int n3)
{
  double (*z)[n2][n1] = ((double (*)[n2][n1])oz);
  int i1;
  int i2;
  int i3;
  int m1;
  int m2;
  int m3;
  m1 = (n1 < 18?n1 : 18);
  m2 = (n2 < 14?n2 : 14);
  m3 = (n3 < 18?n3 : 18);
  printf("   \n");
  for (i3 = 0; i3 <= m3 - 1; i3 += 1) {
    for (i1 = 0; i1 <= m1 - 1; i1 += 1) {
      for (i2 = 0; i2 <= m2 - 1; i2 += 1) {
        printf("%6.3f",z[i3][i2][i1]);
      }
      printf("\n");
    }
    printf("  - - - - - - - \n");
  }
  printf("   \n");
}
//---------------------------------------------------------------------
// power  raises an integer, disguised as a double
// precision real, to an integer power
//---------------------------------------------------------------------

static double power(double a,int n)
{
  double aj;
  int nj;
  double rdummy;
  double power;
  power = 1.0;
  nj = n;
  aj = a;
  while(nj != 0){
    if (nj % 2 == 1) 
      rdummy = randlc(&power,aj);
    rdummy = randlc(&aj,aj);
    nj = nj / 2;
  }
  return power;
}
//---------------------------------------------------------------------
// bubble        does a bubble sort in direction dir
//---------------------------------------------------------------------

static void bubble(double ten[][2],int j1[][2],int j2[][2],int j3[][2],int m,int ind)
{
  double temp;
  int i;
  int j_temp;
  if (ind == 1) {
    for (i = 0; i <= m - 1 - 1; i += 1) {
      if (ten[i][ind] > ten[i + 1][ind]) {
        temp = ten[i + 1][ind];
        ten[i + 1][ind] = ten[i][ind];
        ten[i][ind] = temp;
        j_temp = j1[i + 1][ind];
        j1[i + 1][ind] = j1[i][ind];
        j1[i][ind] = j_temp;
        j_temp = j2[i + 1][ind];
        j2[i + 1][ind] = j2[i][ind];
        j2[i][ind] = j_temp;
        j_temp = j3[i + 1][ind];
        j3[i + 1][ind] = j3[i][ind];
        j3[i][ind] = j_temp;
      }
       else {
        return ;
      }
    }
  }
   else {
    for (i = 0; i <= m - 1 - 1; i += 1) {
      if (ten[i][ind] < ten[i + 1][ind]) {
        temp = ten[i + 1][ind];
        ten[i + 1][ind] = ten[i][ind];
        ten[i][ind] = temp;
        j_temp = j1[i + 1][ind];
        j1[i + 1][ind] = j1[i][ind];
        j1[i][ind] = j_temp;
        j_temp = j2[i + 1][ind];
        j2[i + 1][ind] = j2[i][ind];
        j2[i][ind] = j_temp;
        j_temp = j3[i + 1][ind];
        j3[i + 1][ind] = j3[i][ind];
        j3[i][ind] = j_temp;
      }
       else {
        return ;
      }
    }
  }
}

static void zero3(void *oz,int n1,int n2,int n3)
{
  double (*z)[n2][n1] = ((double (*)[n2][n1])oz);
  int i1;
  int i2;
  int i3;
  
#pragma omp parallel for private (i1,i2,i3) firstprivate (n2,n3)
  for (i3 = 0; i3 <= n3 - 1; i3 += 1) {
    
#pragma omp parallel for private (i1,i2) firstprivate (n1)
    for (i2 = 0; i2 <= n2 - 1; i2 += 1) {
      
#pragma omp parallel for private (i1)
      for (i1 = 0; i1 <= n1 - 1; i1 += 1) {
        z[i3][i2][i1] = 0.0;
      }
    }
  }
}
