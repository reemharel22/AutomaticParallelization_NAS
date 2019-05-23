//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB LU code. This OpenMP  //
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
#include <stdio.h>
#include <math.h>
#include "applu.incl"
//---------------------------------------------------------------------
// 
// compute the solution error
// 
//---------------------------------------------------------------------
#include "omp.h" 

void error()
{
//---------------------------------------------------------------------
// local variables
//---------------------------------------------------------------------
  int i;
  int j;
  int k;
  int m;
  double tmp;
  double u000ijk[5];
  double errnm_local[5];
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 4; m += 1) {
    errnm[m] = 0.0;
  }
{
    
#pragma omp parallel for private (m)
    for (m = 0; m <= 4; m += 1) {
      errnm_local[m] = 0.0;
    }
    for (k = 1; k <= nz - 1 - 1; k += 1) {
      for (j = jst; j <= jend - 1; j += 1) {
        for (i = ist; i <= iend - 1; i += 1) {
          exact(i,j,k,u000ijk);
          
#pragma omp parallel for private (tmp,m)
          for (m = 0; m <= 4; m += 1) {
            tmp = u000ijk[m] - u[k][j][i][m];
            errnm_local[m] = errnm_local[m] + tmp * tmp;
          }
        }
      }
    }
    
#pragma omp parallel for private (m)
    for (m = 0; m <= 4; m += 1) {
      errnm[m] += errnm_local[m];
    }
//end parallel
  }
  for (m = 0; m <= 4; m += 1) {
    errnm[m] = sqrt(errnm[m] / ((nx0 - 2) * (ny0 - 2) * (nz0 - 2)));
  }
/*
  printf(" \n RMS-norm of error in soln. to first pde  = %12.5E\n"
         " RMS-norm of error in soln. to second pde = %12.5E\n"
         " RMS-norm of error in soln. to third pde  = %12.5E\n"
         " RMS-norm of error in soln. to fourth pde = %12.5E\n"
         " RMS-norm of error in soln. to fifth pde  = %12.5E\n",
         errnm[0], errnm[1], errnm[2], errnm[3], errnm[4]);
  */
}
