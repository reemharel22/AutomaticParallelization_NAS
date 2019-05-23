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
#include "applu.incl"
//---------------------------------------------------------------------
// set the boundary values of dependent variables
//---------------------------------------------------------------------
#include "omp.h" 

void setbv()
{
//---------------------------------------------------------------------
// local variables
//---------------------------------------------------------------------
  int i;
  int j;
  int k;
  int m;
  double temp1[5];
  double temp2[5];
//---------------------------------------------------------------------
// set the dependent variable values along the top and bottom faces
//---------------------------------------------------------------------
  for (j = 0; j <= ny - 1; j += 1) {
    for (i = 0; i <= nx - 1; i += 1) {
      exact(i,j,0,temp1);
      exact(i,j,nz - 1,temp2);
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 4; m += 1) {
        u[0][j][i][m] = temp1[m];
        u[nz - 1][j][i][m] = temp2[m];
      }
    }
  }
//---------------------------------------------------------------------
// set the dependent variable values along north and south faces
//---------------------------------------------------------------------
  for (k = 0; k <= nz - 1; k += 1) {
    for (i = 0; i <= nx - 1; i += 1) {
      exact(i,0,k,temp1);
      exact(i,ny - 1,k,temp2);
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 4; m += 1) {
        u[k][0][i][m] = temp1[m];
        u[k][ny - 1][i][m] = temp2[m];
      }
    }
  }
//---------------------------------------------------------------------
// set the dependent variable values along east and west faces
//---------------------------------------------------------------------
  for (k = 0; k <= nz - 1; k += 1) {
    for (j = 0; j <= ny - 1; j += 1) {
      exact(0,j,k,temp1);
      exact(nx - 1,j,k,temp2);
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 4; m += 1) {
        u[k][j][0][m] = temp1[m];
        u[k][j][nx - 1][m] = temp2[m];
      }
    }
  }
}
