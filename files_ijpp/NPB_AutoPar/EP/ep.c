//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB EP code. This OpenMP  //
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
//      program EMBAR
//---------------------------------------------------------------------
//   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//   numbers.  MK is the Log_2 of the size of each batch of uniform random
//   numbers.  MK can be set for convenience on a given system, since it does
//   not affect the results.
//---------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



#include "type.h"
#include "npbparams.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"
#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))
#define MK        16
#define MM        (M - MK)
#define NN        (1 << MM)
#define NK        (1 << MK)
#define NQ        10
#define EPSILON   1.0e-8
#define A         1220703125.0
#define S         271828183.0
#include "omp.h" 
static double x[2 * (1 << 16)];
static double qq[10];
static double q[10];

int main(int argc,char *argv[])
{
  double Mops;
  double t1;
  double t2;
  double t3;
  double t4;
  double x1;
  double x2;
  double sx;
  double sy;
  double tm;
  double an;
  double tt;
  double gc;
  double sx_verify_value;
  double sy_verify_value;
  double sx_err;
  double sy_err;
  int np;
  int i;
  int ik;
  int kk;
  int l;
  int k;
  int nit;
  int k_offset;
  int j;
  logical verified;
  logical timers_enabled;
  double dum[3] = {(1.0), (1.0), (1.0)};
  char size[16];
  FILE *fp;
  if ((fp = fopen("timer.flag","r")) == ((void *)0)) {
    timers_enabled = false;
  }
   else {
    timers_enabled = true;
    fclose(fp);
  }
//--------------------------------------------------------------------
//  Because the size of the problem is too large to store in a 32-bit
//  integer for some classes, we put it into a string (for printing).
//  Have to strip off the decimal point put in there by the floating
//  point print statement (internal file)
//--------------------------------------------------------------------
  sprintf(size,"%15.0lf",(pow(2.0,(32 + 1))));
  j = 14;
  if (size[j] == '.') 
    j--;
  size[j + 1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OMP-C) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n",size);
  printf("\n Number of available threads:          %13d\n",(omp_get_max_threads()));
  verified = false;
//--------------------------------------------------------------------
//  Compute the number of "batches" of random number pairs generated 
//  per processor. Adjust if the number of processors does not evenly 
//  divide the total number
//--------------------------------------------------------------------
  np = 1 << 32 - 16;
printf("np :%d\n", np);
//--------------------------------------------------------------------
//  Call the random number generator functions and initialize
//  the x-array to reduce the effects of paging on the timings.
//  Also, call all mathematical functions that are used. Make
//  sure these initializations cannot be eliminated as dead code.
//--------------------------------------------------------------------
  vranlc(0,&dum[0],dum[1],&dum[2]);
  dum[0] = randlc(&dum[1],dum[2]);

    
#pragma omp parallel for private (i)
    for (i = 0; i <= 131071; i += 1) {
      x[i] = - 1.0e99;
    }
  
  Mops = log((sqrt((fabs((1.0 > 1.0?1.0 : 1.0))))));

    timer_clear(0);
    if (timers_enabled) 
      timer_clear(1);
    if (timers_enabled) 
      timer_clear(2);

  timer_start(0);
  t1 = 1220703125.0;
  vranlc(0,&t1,1220703125.0,x);
//--------------------------------------------------------------------
//  Compute AN = A ^ (2 * NK) (mod 2^46).
//--------------------------------------------------------------------
  t1 = 1220703125.0;
  for (i = 0; i <= 16; i += 1) {
    t2 = randlc(&t1,t1);
  }
  an = t1;
  tt = 271828183.0;
  gc = 0.0;
  sx = 0.0;
  sy = 0.0;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 9; i += 1) {
    q[i] = 0.0;
  }
//--------------------------------------------------------------------
//  Each instance of this loop may be performed independently. We compute
//  the k offsets separately to take into account the fact that some nodes
//  have more numbers to generate than others
//--------------------------------------------------------------------
  k_offset = - 1;

    
#pragma omp parallel for private (i)
    for (i = 0; i <= 9; i += 1) {
      qq[i] = 0.0;
    }
    for (k = 1; k <= np; k += 1) {
      kk = k_offset + k;
      t1 = 271828183.0;
      t2 = an;
// Find starting seed t1 for this kk.
      for (i = 1; i <= 100; i += 1) {
        ik = kk / 2;
        if (2 * ik != kk) 
          t3 = randlc(&t1,t2);
        if (ik == 0) 
          break; 
        t3 = randlc(&t2,t2);
        kk = ik;
      }
//--------------------------------------------------------------------
//  Compute uniform pseudorandom numbers.
//--------------------------------------------------------------------
      if (timers_enabled) 
        timer_start(2);
      vranlc(2 * (1 << 16),&t1,1220703125.0,x);
      if (timers_enabled) 
        timer_stop(2);
//--------------------------------------------------------------------
//  Compute Gaussian deviates by acceptance-rejection method and 
//  tally counts in concentri//square annuli.  This loop is not 
//  vectorizable. 
//--------------------------------------------------------------------
      if (timers_enabled) 
        timer_start(1);
      for (i = 0; i <= 65535; i += 1) {
        x1 = 2.0 * x[2 * i] - 1.0;
        x2 = 2.0 * x[2 * i + 1] - 1.0;
        t1 = x1 * x1 + x2 * x2;
        if (t1 <= 1.0) {
          t2 = sqrt(- 2.0 * log(t1) / t1);
          t3 = x1 * t2;
          t4 = x2 * t2;
          l = ((fabs(t3) > fabs(t4)?fabs(t3) : fabs(t4)));
          qq[l] = qq[l] + 1.0;
          sx = sx + t3;
          sy = sy + t4;
        }
      }
      if (timers_enabled) 
        timer_stop(1);
    }
    
#pragma omp parallel for private (i)
    for (i = 0; i <= 9; i += 1) {
      q[i] += qq[i];
    }
  
  
#pragma omp parallel for private (i) reduction (+:gc)
  for (i = 0; i <= 9; i += 1) {
    gc = gc + q[i];
  }
  timer_stop(0);
  tm = timer_read(0);
  nit = 0;
  verified = true;
  if (32 == 24) {
    sx_verify_value = - 3.247834652034740e+3;
    sy_verify_value = - 6.958407078382297e+3;
  }
   else if (32 == 25) {
    sx_verify_value = - 2.863319731645753e+3;
    sy_verify_value = - 6.320053679109499e+3;
  }
   else if (32 == 28) {
    sx_verify_value = - 4.295875165629892e+3;
    sy_verify_value = - 1.580732573678431e+4;
  }
   else if (32 == 30) {
    sx_verify_value = 4.033815542441498e+4;
    sy_verify_value = - 2.660669192809235e+4;
  }
   else if (32 == 32) {
    sx_verify_value = 4.764367927995374e+4;
    sy_verify_value = - 8.084072988043731e+4;
  }
   else if (32 == 36) {
    sx_verify_value = 1.982481200946593e+5;
    sy_verify_value = - 1.020596636361769e+5;
  }
   else if (32 == 40) {
    sx_verify_value = - 5.319717441530e+05;
    sy_verify_value = - 3.688834557731e+05;
  }
   else {
    verified = false;
  }
  if (verified) {
    sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
    sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
    verified = (sx_err <= 1.0e-8 && sy_err <= 1.0e-8);
  }
  Mops = pow(2.0,(32 + 1)) / tm / 1000000.0;
  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n",tm);
  printf("N = 2^%5d\n",32);
  printf("No. Gaussian Pairs = %15.0lf\n",gc);
  printf("Sums = %25.15lE %25.15lE\n",sx,sy);
  printf("Counts: \n");
  for (i = 0; i <= 9; i += 1) {
    printf("%3d%15.0lf\n",i,q[i]);
  }
  print_results("EP",'C',32 + 1,0,0,nit,tm,Mops,"Random numbers generated",verified,"3.3.1","23 Oct 2018","icc","$(CC)","-lm","-I../common","-g -Wall -O3 -qopenmp -mcmodel=medium","-O3 -qopenmp -mcmodel=medium","randdp");
  if (timers_enabled) {
    if (tm <= 0.0) 
      tm = 1.0;
    tt = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n",tt,tt * 100.0 / tm);
    tt = timer_read(1);
    printf("Gaussian pairs: %9.3lf (%6.2lf)\n",tt,tt * 100.0 / tm);
    tt = timer_read(2);
    printf("Random numbers: %9.3lf (%6.2lf)\n",tt,tt * 100.0 / tm);
  }
  return 0;
}
