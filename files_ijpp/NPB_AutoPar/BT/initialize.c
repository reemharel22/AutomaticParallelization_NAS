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
#include "header.h"
//---------------------------------------------------------------------
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
#include "omp.h" 

void initialize()
{
  int i;
  int j;
  int k;
  int m;
  int ix;
  int iy;
  int iz;
  double xi;
  double eta;
  double zeta;
  double Pface[2][3][5];
  double Pxi;
  double Peta;
  double Pzeta;
  double temp[5];
//---------------------------------------------------------------------
// Later (in compute_rhs) we compute 1/u for every element. A few of 
// the corner elements are not used, but it convenient (and faster) 
// to compute the whole thing with a simple loop. Make sure those 
// values are nonzero by initializing the whole thing here. 
//---------------------------------------------------------------------
  for (k = 0; k <= grid_points[2] - 1; k += 1) {
    
#pragma omp parallel for private (i,j,m)
    for (j = 0; j <= grid_points[1] - 1; j += 1) {
      
#pragma omp parallel for private (i,m)
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        
#pragma omp parallel for private (m) firstprivate (k)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = 1.0;
        }
      }
    }
//---------------------------------------------------------------------
// first store the "interpolated" values everywhere on the grid    
//---------------------------------------------------------------------
    for (k = 0; k <= grid_points[2] - 1; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (j = 0; j <= grid_points[1] - 1; j += 1) {
        eta = ((double )j) * dnym1;
        for (i = 0; i <= grid_points[0] - 1; i += 1) {
          xi = ((double )i) * dnxm1;
          for (ix = 0; ix <= 1; ix += 1) {
            exact_solution(((double )ix),eta,zeta,&Pface[ix][0][0]);
          }
          for (iy = 0; iy <= 1; iy += 1) {
            exact_solution(xi,((double )iy),zeta,&Pface[iy][1][0]);
          }
          for (iz = 0; iz <= 1; iz += 1) {
            exact_solution(xi,eta,((double )iz),&Pface[iz][2][0]);
          }
          
#pragma omp parallel for private (Pxi,Peta,Pzeta,m) firstprivate (xi,eta,zeta)
          for (m = 0; m <= 4; m += 1) {
            Pxi = xi * Pface[1][0][m] + (1.0 - xi) * Pface[0][0][m];
            Peta = eta * Pface[1][1][m] + (1.0 - eta) * Pface[0][1][m];
            Pzeta = zeta * Pface[1][2][m] + (1.0 - zeta) * Pface[0][2][m];
            u[k][j][i][m] = Pxi + Peta + Pzeta - Pxi * Peta - Pxi * Pzeta - Peta * Pzeta + Pxi * Peta * Pzeta;
          }
        }
      }
    }
//---------------------------------------------------------------------
// now store the exact values on the boundaries        
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// west face                                                  
//---------------------------------------------------------------------
    i = 0;
    xi = 0.0;
    for (k = 0; k <= grid_points[2] - 1; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (j = 0; j <= grid_points[1] - 1; j += 1) {
        eta = ((double )j) * dnym1;
        exact_solution(xi,eta,zeta,temp);
        
#pragma omp parallel for private (m) firstprivate (i)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = temp[m];
        }
      }
    }
//---------------------------------------------------------------------
// east face                                                      
//---------------------------------------------------------------------
    i = grid_points[0] - 1;
    xi = 1.0;
    for (k = 0; k <= grid_points[2] - 1; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (j = 0; j <= grid_points[1] - 1; j += 1) {
        eta = ((double )j) * dnym1;
        exact_solution(xi,eta,zeta,temp);
        
#pragma omp parallel for private (m) firstprivate (i)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = temp[m];
        }
      }
    }
//---------------------------------------------------------------------
// south face                                                 
//---------------------------------------------------------------------
    j = 0;
    eta = 0.0;
    for (k = 0; k <= grid_points[2] - 1; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        xi = ((double )i) * dnxm1;
        exact_solution(xi,eta,zeta,temp);
        
#pragma omp parallel for private (m) firstprivate (j)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = temp[m];
        }
      }
    }
//---------------------------------------------------------------------
// north face                                    
//---------------------------------------------------------------------
    j = grid_points[1] - 1;
    eta = 1.0;
    for (k = 0; k <= grid_points[2] - 1; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        xi = ((double )i) * dnxm1;
        exact_solution(xi,eta,zeta,temp);
        
#pragma omp parallel for private (m) firstprivate (j)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = temp[m];
        }
      }
    }
//---------------------------------------------------------------------
// bottom face                                       
//---------------------------------------------------------------------
    k = 0;
    zeta = 0.0;
    for (j = 0; j <= grid_points[1] - 1; j += 1) {
      eta = ((double )j) * dnym1;
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        xi = ((double )i) * dnxm1;
        exact_solution(xi,eta,zeta,temp);
        
#pragma omp parallel for private (m) firstprivate (k)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = temp[m];
        }
      }
    }
//---------------------------------------------------------------------
// top face     
//---------------------------------------------------------------------
    k = grid_points[2] - 1;
    zeta = 1.0;
    for (j = 0; j <= grid_points[1] - 1; j += 1) {
      eta = ((double )j) * dnym1;
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        xi = ((double )i) * dnxm1;
        exact_solution(xi,eta,zeta,temp);
        
#pragma omp parallel for private (m)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = temp[m];
        }
      }
    }
//end parallel
  }
}

void lhsinit(double lhs[][3][5][5],int ni)
{
  int i;
  int m;
  int n;
//---------------------------------------------------------------------
// zero the whole left hand side for starters
// set all diagonal values to 1. This is overkill, but convenient
//---------------------------------------------------------------------
  i = 0;
  
#pragma omp parallel for private (m,n) firstprivate (i)
  for (n = 0; n <= 4; n += 1) {
    
#pragma omp parallel for private (m)
    for (m = 0; m <= 4; m += 1) {
      lhs[i][0][n][m] = 0.0;
      lhs[i][1][n][m] = 0.0;
      lhs[i][2][n][m] = 0.0;
    }
    lhs[i][1][n][n] = 1.0;
  }
  i = ni;
  
#pragma omp parallel for private (m,n) firstprivate (i)
  for (n = 0; n <= 4; n += 1) {
    
#pragma omp parallel for private (m)
    for (m = 0; m <= 4; m += 1) {
      lhs[i][0][n][m] = 0.0;
      lhs[i][1][n][m] = 0.0;
      lhs[i][2][n][m] = 0.0;
    }
    lhs[i][1][n][n] = 1.0;
  }
}
