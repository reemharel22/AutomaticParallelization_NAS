//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB UA code. This OpenMP  //
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
#include <math.h>
#include "header.h"
#include "timers.h"
//---------------------------------------------------------
// Advance the convection term using 4th order RK
// 1.ta1 is solution from last time step 
// 2.the heat source is considered part of d/dx
// 3.trhs is right hand side for the diffusion equation
// 4.tmor is solution on mortar points, which will be used
//   as the initial guess when advancing the diffusion term 
//---------------------------------------------------------
#include "omp.h" 

void convect(logical ifmortar)
{
  double alpha2;
  double tempa[5][5][5];
  double rdtime;
  double pidivalpha;
  double dtx1;
  double dtx2;
  double dtx3;
  double src;
  double rk1[5][5][5];
  double rk2[5][5][5];
  double rk3[5][5][5];
  double rk4[5][5][5];
  double temp[5][5][5];
  double subtime[3];
  double xx0[3];
  double yy0[3];
  double zz0[3];
  double dtime2;
  double r2;
  double sum;
  double xloc[5];
  double yloc[5];
  double zloc[5];
  int k;
  int iel;
  int i;
  int j;
  int iside;
  int isize;
  int substep;
  int ip;
  const double sixth = 1.0 / 6.0;
  if (timeron) 
    timer_start(3);
  pidivalpha = acos(- 1.0) / alpha;
  alpha2 = alpha * alpha;
  dtime2 = dtime / 2.0;
  rdtime = 1.0 / dtime;
  subtime[0] = time;
  subtime[1] = time + dtime2;
  subtime[2] = time + dtime;
  
#pragma omp parallel for private (substep)
  for (substep = 0; substep <= 2; substep += 1) {
    xx0[substep] = 3.0 / 7.0 + 3.0 * subtime[substep];
    yy0[substep] = 2.0 / 7.0 + 3.0 * subtime[substep];
    zz0[substep] = 2.0 / 7.0 + 3.0 * subtime[substep];
  }
  for (iel = 0; iel <= nelt - 1; iel += 1) {
    isize = size_e[iel];
/*
    xloc[i] is the location of i'th collocation in x direction in an element.
    yloc[i] is the location of j'th collocation in y direction in an element.
    zloc[i] is the location of k'th collocation in z direction in an element.
    */
    
#pragma omp parallel for private (i)
    for (i = 0; i <= 4; i += 1) {
      xloc[i] = xfrac[i] * (xc[iel][1] - xc[iel][0]) + xc[iel][0];
    }
    
#pragma omp parallel for private (j)
    for (j = 0; j <= 4; j += 1) {
      yloc[j] = xfrac[j] * (yc[iel][3] - yc[iel][0]) + yc[iel][0];
    }
    
#pragma omp parallel for private (k)
    for (k = 0; k <= 4; k += 1) {
      zloc[k] = xfrac[k] * (zc[iel][4] - zc[iel][0]) + zc[iel][0];
    }
    for (k = 0; k <= 4; k += 1) {
      for (j = 0; j <= 4; j += 1) {
        for (i = 0; i <= 4; i += 1) {
          r2 = pow(xloc[i] - xx0[0],2.0) + pow(yloc[j] - yy0[0],2.0) + pow(zloc[k] - zz0[0],2.0);
          if (r2 <= alpha2) {
            src = cos(sqrt(r2) * pidivalpha) + 1.0;
          }
           else {
            src = 0.0;
          }
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][i] * ta1[iel][k][j][ip];
          }
          dtx1 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][j] * ta1[iel][k][ip][i];
          }
          dtx2 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][k] * ta1[iel][ip][j][i];
          }
          dtx3 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          rk1[k][j][i] = dtx1 + dtx2 + dtx3 + src;
          temp[k][j][i] = ta1[iel][k][j][i] + dtime2 * rk1[k][j][i];
        }
      }
    }
    for (k = 0; k <= 4; k += 1) {
      for (j = 0; j <= 4; j += 1) {
        for (i = 0; i <= 4; i += 1) {
          r2 = pow(xloc[i] - xx0[1],2.0) + pow(yloc[j] - yy0[1],2.0) + pow(zloc[k] - zz0[1],2.0);
          if (r2 <= alpha2) {
            src = cos(sqrt(r2) * pidivalpha) + 1.0;
          }
           else {
            src = 0.0;
          }
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][i] * temp[k][j][ip];
          }
          dtx1 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][j] * temp[k][ip][i];
          }
          dtx2 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][k] * temp[ip][j][i];
          }
          dtx3 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          rk2[k][j][i] = dtx1 + dtx2 + dtx3 + src;
          tempa[k][j][i] = ta1[iel][k][j][i] + dtime2 * rk2[k][j][i];
        }
      }
    }
    for (k = 0; k <= 4; k += 1) {
      for (j = 0; j <= 4; j += 1) {
        for (i = 0; i <= 4; i += 1) {
          r2 = pow(xloc[i] - xx0[1],2.0) + pow(yloc[j] - yy0[1],2.0) + pow(zloc[k] - zz0[1],2.0);
          if (r2 <= alpha2) {
            src = cos(sqrt(r2) * pidivalpha) + 1.0;
          }
           else {
            src = 0.0;
          }
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][i] * tempa[k][j][ip];
          }
          dtx1 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][j] * tempa[k][ip][i];
          }
          dtx2 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][k] * tempa[ip][j][i];
          }
          dtx3 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          rk3[k][j][i] = dtx1 + dtx2 + dtx3 + src;
          temp[k][j][i] = ta1[iel][k][j][i] + dtime * rk3[k][j][i];
        }
      }
    }
    for (k = 0; k <= 4; k += 1) {
      for (j = 0; j <= 4; j += 1) {
        for (i = 0; i <= 4; i += 1) {
          r2 = pow(xloc[i] - xx0[2],2.0) + pow(yloc[j] - yy0[2],2.0) + pow(zloc[k] - zz0[2],2.0);
          if (r2 <= alpha2) {
            src = cos(sqrt(r2) * pidivalpha) + 1.0;
          }
           else {
            src = 0.0;
          }
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][i] * temp[k][j][ip];
          }
          dtx1 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][j] * temp[k][ip][i];
          }
          dtx2 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          sum = 0.0;
          
#pragma omp parallel for private (ip) reduction (+:sum)
          for (ip = 0; ip <= 4; ip += 1) {
            sum = sum + dxm1[ip][k] * temp[ip][j][i];
          }
          dtx3 = - 3.0 * sum * xrm1_s[isize][k][j][i];
          rk4[k][j][i] = dtx1 + dtx2 + dtx3 + src;
          tempa[k][j][i] = sixth * (rk1[k][j][i] + 2.0 * rk2[k][j][i] + 2.0 * rk3[k][j][i] + rk4[k][j][i]);
        }
      }
    }
// apply boundary condition
    for (iside = 0; iside <= 5; iside += 1) {
      if (cbc[iel][iside] == 0) {
        facev(tempa,iside,0.0);
      }
    }
    
#pragma omp parallel for private (k,i,j)
    for (k = 0; k <= 4; k += 1) {
      
#pragma omp parallel for private (i,j)
      for (j = 0; j <= 4; j += 1) {
        
#pragma omp parallel for private (i) firstprivate (dtime,rdtime,isize)
        for (i = 0; i <= 4; i += 1) {
          trhs[iel][k][j][i] = bm1_s[isize][k][j][i] * (ta1[iel][k][j][i] * rdtime + tempa[k][j][i]);
          ta1[iel][k][j][i] = ta1[iel][k][j][i] + tempa[k][j][i] * dtime;
        }
      }
    }
  }
// get mortar for intial guess for CG
  if (timeron) 
    timer_start(4);
  if (ifmortar) {
    transfb_c_2(((double *)ta1));
  }
   else {
    transfb_c(((double *)ta1));
  }
  if (timeron) 
    timer_stop(4);
  
#pragma omp parallel for private (i) firstprivate (nmor)
  for (i = 0; i <= nmor - 1; i += 1) {
    tmort[i] = tmort[i] / mormult[i];
  }
  if (timeron) 
    timer_stop(3);
}
