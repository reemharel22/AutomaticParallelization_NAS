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
#include <math.h>
#include "header.h"
//---------------------------------------------------------------------
// this function computes the norm of the difference between the
// computed solution and the exact solution
//---------------------------------------------------------------------
#include "omp.h" 

void error_norm(double rms[5])
{
  int i;
  int j;
  int k;
  int m;
  int d;
  double xi;
  double eta;
  double zeta;
  double u_exact[5];
  double add;
  double rms_local[5];
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    rms[m] = 0.0;
  }
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    rms_local[m] = 0.0;
  }
  for (k = 0; k <= grid_points[2] - 1; k += 1) {
    zeta = ((double )k) * dnzm1;
    for (j = 0; j <= grid_points[1] - 1; j += 1) {
      eta = ((double )j) * dnym1;
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        xi = ((double )i) * dnxm1;
        exact_solution(xi,eta,zeta,u_exact);
        
#pragma omp parallel for private (add,m)
        for (m = 0; m <= 4; m += 1) {
          add = u[k][j][i][m] - u_exact[m];
          rms_local[m] = rms_local[m] + add * add;
        }
      }
    }
  }
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    rms[m] += rms_local[m];
  }
  for (m = 0; m <= 4; m += 1) {
    for (d = 0; d <= 2; d += 1) {
      rms[m] = rms[m] / ((double )(grid_points[d] - 2));
    }
    rms[m] = sqrt(rms[m]);
  }
}

void rhs_norm(double rms[5])
{
  int i;
  int j;
  int k;
  int d;
  int m;
  double add;
  double rms_local[5];
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    rms[m] = 0.0;
  }
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    rms_local[m] = 0.0;
  }
  for (k = 1; k <= grid_points[2] - 2; k += 1) {
    for (j = 1; j <= grid_points[1] - 2; j += 1) {
      for (i = 1; i <= grid_points[0] - 2; i += 1) {
        
#pragma omp parallel for private (add,m)
        for (m = 0; m <= 4; m += 1) {
          add = rhs[k][j][i][m];
          rms_local[m] = rms_local[m] + add * add;
        }
      }
    }
  }
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    rms[m] += rms_local[m];
  }
  for (m = 0; m <= 4; m += 1) {
    for (d = 0; d <= 2; d += 1) {
      rms[m] = rms[m] / ((double )(grid_points[d] - 2));
    }
    rms[m] = sqrt(rms[m]);
  }
}
