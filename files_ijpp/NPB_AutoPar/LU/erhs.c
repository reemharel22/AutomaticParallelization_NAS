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
//
// compute the right hand side based on exact solution
//
//---------------------------------------------------------------------
#include "omp.h" 

void erhs()
{
//---------------------------------------------------------------------
// local variables
//---------------------------------------------------------------------
  int i;
  int j;
  int k;
  int m;
  double xi;
  double eta;
  double zeta;
  double q;
  double u21;
  double u31;
  double u41;
  double tmp;
  double u21i;
  double u31i;
  double u41i;
  double u51i;
  double u21j;
  double u31j;
  double u41j;
  double u51j;
  double u21k;
  double u31k;
  double u41k;
  double u51k;
  double u21im1;
  double u31im1;
  double u41im1;
  double u51im1;
  double u21jm1;
  double u31jm1;
  double u41jm1;
  double u51jm1;
  double u21km1;
  double u31km1;
  double u41km1;
  double u51km1;
  
#pragma omp parallel for private (i,j,k,m)
  for (k = 0; k <= nz - 1; k += 1) {
    
#pragma omp parallel for private (i,j,m) firstprivate (nx)
    for (j = 0; j <= ny - 1; j += 1) {
      
#pragma omp parallel for private (i,m)
      for (i = 0; i <= nx - 1; i += 1) {
        
#pragma omp parallel for private (m)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = 0.0;
        }
      }
    }
  }
  
#pragma omp parallel for private (xi,eta,zeta,i,j,k,m) firstprivate (ny)
  for (k = 0; k <= nz - 1; k += 1) {
    zeta = ((double )k) / (nz - 1);
    
#pragma omp parallel for private (xi,eta,i,j,m) firstprivate (nx,ny0)
    for (j = 0; j <= ny - 1; j += 1) {
      eta = ((double )j) / (ny0 - 1);
      
#pragma omp parallel for private (xi,i,m) firstprivate (nx0)
      for (i = 0; i <= nx - 1; i += 1) {
        xi = ((double )i) / (nx0 - 1);
        
#pragma omp parallel for private (m) firstprivate (xi,eta,zeta)
        for (m = 0; m <= 4; m += 1) {
          rsd[k][j][i][m] = ce[m][0] + (ce[m][1] + (ce[m][4] + (ce[m][7] + ce[m][10] * xi) * xi) * xi) * xi + (ce[m][2] + (ce[m][5] + (ce[m][8] + ce[m][11] * eta) * eta) * eta) * eta + (ce[m][3] + (ce[m][6] + (ce[m][9] + ce[m][12] * zeta) * zeta) * zeta) * zeta;
        }
      }
    }
  }
//---------------------------------------------------------------------
// xi-direction flux differences
//---------------------------------------------------------------------
  for (k = 1; k <= nz - 1 - 1; k += 1) {
    for (j = jst; j <= jend - 1; j += 1) {
      
#pragma omp parallel for private (q,u21,i)
      for (i = 0; i <= nx - 1; i += 1) {
        flux[i][0] = rsd[k][j][i][1];
        u21 = rsd[k][j][i][1] / rsd[k][j][i][0];
        q = 0.50 * (rsd[k][j][i][1] * rsd[k][j][i][1] + rsd[k][j][i][2] * rsd[k][j][i][2] + rsd[k][j][i][3] * rsd[k][j][i][3]) / rsd[k][j][i][0];
        flux[i][1] = rsd[k][j][i][1] * u21 + 0.40e+00 * (rsd[k][j][i][4] - q);
        flux[i][2] = rsd[k][j][i][2] * u21;
        flux[i][3] = rsd[k][j][i][3] * u21;
        flux[i][4] = (1.40e+00 * rsd[k][j][i][4] - 0.40e+00 * q) * u21;
      }
      
#pragma omp parallel for private (i,m)
      for (i = ist; i <= iend - 1; i += 1) {
        
#pragma omp parallel for private (m) firstprivate (tx2)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = frct[k][j][i][m] - tx2 * (flux[i + 1][m] - flux[i - 1][m]);
        }
      }
      
#pragma omp parallel for private (tmp,u21i,u31i,u41i,u51i,u21im1,u31im1,u41im1,u51im1,i)
      for (i = ist; i <= nx - 1; i += 1) {
        tmp = 1.0 / rsd[k][j][i][0];
        u21i = tmp * rsd[k][j][i][1];
        u31i = tmp * rsd[k][j][i][2];
        u41i = tmp * rsd[k][j][i][3];
        u51i = tmp * rsd[k][j][i][4];
        tmp = 1.0 / rsd[k][j][i - 1][0];
        u21im1 = tmp * rsd[k][j][i - 1][1];
        u31im1 = tmp * rsd[k][j][i - 1][2];
        u41im1 = tmp * rsd[k][j][i - 1][3];
        u51im1 = tmp * rsd[k][j][i - 1][4];
        flux[i][1] = 4.0 / 3.0 * tx3 * (u21i - u21im1);
        flux[i][2] = tx3 * (u31i - u31im1);
        flux[i][3] = tx3 * (u41i - u41im1);
        flux[i][4] = 0.50 * (1.0 - 1.40e+00 * 1.40e+00) * tx3 * (u21i * u21i + u31i * u31i + u41i * u41i - (u21im1 * u21im1 + u31im1 * u31im1 + u41im1 * u41im1)) + 1.0 / 6.0 * tx3 * (u21i * u21i - u21im1 * u21im1) + 1.40e+00 * 1.40e+00 * tx3 * (u51i - u51im1);
      }
      
#pragma omp parallel for private (i) firstprivate (tx1,tx3,dx1,dx2,dx3,dx4,dx5)
      for (i = ist; i <= iend - 1; i += 1) {
        frct[k][j][i][0] = frct[k][j][i][0] + dx1 * tx1 * (rsd[k][j][i - 1][0] - 2.0 * rsd[k][j][i][0] + rsd[k][j][i + 1][0]);
        frct[k][j][i][1] = frct[k][j][i][1] + tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][1] - flux[i][1]) + dx2 * tx1 * (rsd[k][j][i - 1][1] - 2.0 * rsd[k][j][i][1] + rsd[k][j][i + 1][1]);
        frct[k][j][i][2] = frct[k][j][i][2] + tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][2] - flux[i][2]) + dx3 * tx1 * (rsd[k][j][i - 1][2] - 2.0 * rsd[k][j][i][2] + rsd[k][j][i + 1][2]);
        frct[k][j][i][3] = frct[k][j][i][3] + tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][3] - flux[i][3]) + dx4 * tx1 * (rsd[k][j][i - 1][3] - 2.0 * rsd[k][j][i][3] + rsd[k][j][i + 1][3]);
        frct[k][j][i][4] = frct[k][j][i][4] + tx3 * 1.00e-01 * 1.00e+00 * (flux[i + 1][4] - flux[i][4]) + dx5 * tx1 * (rsd[k][j][i - 1][4] - 2.0 * rsd[k][j][i][4] + rsd[k][j][i + 1][4]);
      }
//---------------------------------------------------------------------
// Fourth-order dissipation
//---------------------------------------------------------------------
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 4; m += 1) {
        frct[k][j][1][m] = frct[k][j][1][m] - dssp * (+5.0 * rsd[k][j][1][m] - 4.0 * rsd[k][j][2][m] + rsd[k][j][3][m]);
        frct[k][j][2][m] = frct[k][j][2][m] - dssp * (- 4.0 * rsd[k][j][1][m] + 6.0 * rsd[k][j][2][m] - 4.0 * rsd[k][j][3][m] + rsd[k][j][4][m]);
      }
      
#pragma omp parallel for private (i,m)
      for (i = 3; i <= nx - 3 - 1; i += 1) {
        
#pragma omp parallel for private (m)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = frct[k][j][i][m] - dssp * (rsd[k][j][i - 2][m] - 4.0 * rsd[k][j][i - 1][m] + 6.0 * rsd[k][j][i][m] - 4.0 * rsd[k][j][i + 1][m] + rsd[k][j][i + 2][m]);
        }
      }
      
#pragma omp parallel for private (m) firstprivate (dssp)
      for (m = 0; m <= 4; m += 1) {
        frct[k][j][nx - 3][m] = frct[k][j][nx - 3][m] - dssp * (rsd[k][j][nx - 5][m] - 4.0 * rsd[k][j][nx - 4][m] + 6.0 * rsd[k][j][nx - 3][m] - 4.0 * rsd[k][j][nx - 2][m]);
        frct[k][j][nx - 2][m] = frct[k][j][nx - 2][m] - dssp * (rsd[k][j][nx - 4][m] - 4.0 * rsd[k][j][nx - 3][m] + 5.0 * rsd[k][j][nx - 2][m]);
      }
    }
  }
//---------------------------------------------------------------------
// eta-direction flux differences
//---------------------------------------------------------------------
  for (k = 1; k <= nz - 1 - 1; k += 1) {
    for (i = ist; i <= iend - 1; i += 1) {
      
#pragma omp parallel for private (q,u31,j)
      for (j = 0; j <= ny - 1; j += 1) {
        flux[j][0] = rsd[k][j][i][2];
        u31 = rsd[k][j][i][2] / rsd[k][j][i][0];
        q = 0.50 * (rsd[k][j][i][1] * rsd[k][j][i][1] + rsd[k][j][i][2] * rsd[k][j][i][2] + rsd[k][j][i][3] * rsd[k][j][i][3]) / rsd[k][j][i][0];
        flux[j][1] = rsd[k][j][i][1] * u31;
        flux[j][2] = rsd[k][j][i][2] * u31 + 0.40e+00 * (rsd[k][j][i][4] - q);
        flux[j][3] = rsd[k][j][i][3] * u31;
        flux[j][4] = (1.40e+00 * rsd[k][j][i][4] - 0.40e+00 * q) * u31;
      }
      
#pragma omp parallel for private (j,m)
      for (j = jst; j <= jend - 1; j += 1) {
        
#pragma omp parallel for private (m) firstprivate (ty2)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = frct[k][j][i][m] - ty2 * (flux[j + 1][m] - flux[j - 1][m]);
        }
      }
      
#pragma omp parallel for private (tmp,u21j,u31j,u41j,u51j,u21jm1,u31jm1,u41jm1,u51jm1,j)
      for (j = jst; j <= ny - 1; j += 1) {
        tmp = 1.0 / rsd[k][j][i][0];
        u21j = tmp * rsd[k][j][i][1];
        u31j = tmp * rsd[k][j][i][2];
        u41j = tmp * rsd[k][j][i][3];
        u51j = tmp * rsd[k][j][i][4];
        tmp = 1.0 / rsd[k][j - 1][i][0];
        u21jm1 = tmp * rsd[k][j - 1][i][1];
        u31jm1 = tmp * rsd[k][j - 1][i][2];
        u41jm1 = tmp * rsd[k][j - 1][i][3];
        u51jm1 = tmp * rsd[k][j - 1][i][4];
        flux[j][1] = ty3 * (u21j - u21jm1);
        flux[j][2] = 4.0 / 3.0 * ty3 * (u31j - u31jm1);
        flux[j][3] = ty3 * (u41j - u41jm1);
        flux[j][4] = 0.50 * (1.0 - 1.40e+00 * 1.40e+00) * ty3 * (u21j * u21j + u31j * u31j + u41j * u41j - (u21jm1 * u21jm1 + u31jm1 * u31jm1 + u41jm1 * u41jm1)) + 1.0 / 6.0 * ty3 * (u31j * u31j - u31jm1 * u31jm1) + 1.40e+00 * 1.40e+00 * ty3 * (u51j - u51jm1);
      }
      
#pragma omp parallel for private (j) firstprivate (ty1,ty3,dy1,dy2,dy3,dy4,dy5)
      for (j = jst; j <= jend - 1; j += 1) {
        frct[k][j][i][0] = frct[k][j][i][0] + dy1 * ty1 * (rsd[k][j - 1][i][0] - 2.0 * rsd[k][j][i][0] + rsd[k][j + 1][i][0]);
        frct[k][j][i][1] = frct[k][j][i][1] + ty3 * 1.00e-01 * 1.00e+00 * (flux[j + 1][1] - flux[j][1]) + dy2 * ty1 * (rsd[k][j - 1][i][1] - 2.0 * rsd[k][j][i][1] + rsd[k][j + 1][i][1]);
        frct[k][j][i][2] = frct[k][j][i][2] + ty3 * 1.00e-01 * 1.00e+00 * (flux[j + 1][2] - flux[j][2]) + dy3 * ty1 * (rsd[k][j - 1][i][2] - 2.0 * rsd[k][j][i][2] + rsd[k][j + 1][i][2]);
        frct[k][j][i][3] = frct[k][j][i][3] + ty3 * 1.00e-01 * 1.00e+00 * (flux[j + 1][3] - flux[j][3]) + dy4 * ty1 * (rsd[k][j - 1][i][3] - 2.0 * rsd[k][j][i][3] + rsd[k][j + 1][i][3]);
        frct[k][j][i][4] = frct[k][j][i][4] + ty3 * 1.00e-01 * 1.00e+00 * (flux[j + 1][4] - flux[j][4]) + dy5 * ty1 * (rsd[k][j - 1][i][4] - 2.0 * rsd[k][j][i][4] + rsd[k][j + 1][i][4]);
      }
//---------------------------------------------------------------------
// fourth-order dissipation
//---------------------------------------------------------------------
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 4; m += 1) {
        frct[k][1][i][m] = frct[k][1][i][m] - dssp * (+5.0 * rsd[k][1][i][m] - 4.0 * rsd[k][2][i][m] + rsd[k][3][i][m]);
        frct[k][2][i][m] = frct[k][2][i][m] - dssp * (- 4.0 * rsd[k][1][i][m] + 6.0 * rsd[k][2][i][m] - 4.0 * rsd[k][3][i][m] + rsd[k][4][i][m]);
      }
      
#pragma omp parallel for private (j,m)
      for (j = 3; j <= ny - 3 - 1; j += 1) {
        
#pragma omp parallel for private (m)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = frct[k][j][i][m] - dssp * (rsd[k][j - 2][i][m] - 4.0 * rsd[k][j - 1][i][m] + 6.0 * rsd[k][j][i][m] - 4.0 * rsd[k][j + 1][i][m] + rsd[k][j + 2][i][m]);
        }
      }
      
#pragma omp parallel for private (m) firstprivate (dssp)
      for (m = 0; m <= 4; m += 1) {
        frct[k][ny - 3][i][m] = frct[k][ny - 3][i][m] - dssp * (rsd[k][ny - 5][i][m] - 4.0 * rsd[k][ny - 4][i][m] + 6.0 * rsd[k][ny - 3][i][m] - 4.0 * rsd[k][ny - 2][i][m]);
        frct[k][ny - 2][i][m] = frct[k][ny - 2][i][m] - dssp * (rsd[k][ny - 4][i][m] - 4.0 * rsd[k][ny - 3][i][m] + 5.0 * rsd[k][ny - 2][i][m]);
      }
    }
  }
//---------------------------------------------------------------------
// zeta-direction flux differences
//---------------------------------------------------------------------
  for (j = jst; j <= jend - 1; j += 1) {
    for (i = ist; i <= iend - 1; i += 1) {
      
#pragma omp parallel for private (q,u41,k)
      for (k = 0; k <= nz - 1; k += 1) {
        flux[k][0] = rsd[k][j][i][3];
        u41 = rsd[k][j][i][3] / rsd[k][j][i][0];
        q = 0.50 * (rsd[k][j][i][1] * rsd[k][j][i][1] + rsd[k][j][i][2] * rsd[k][j][i][2] + rsd[k][j][i][3] * rsd[k][j][i][3]) / rsd[k][j][i][0];
        flux[k][1] = rsd[k][j][i][1] * u41;
        flux[k][2] = rsd[k][j][i][2] * u41;
        flux[k][3] = rsd[k][j][i][3] * u41 + 0.40e+00 * (rsd[k][j][i][4] - q);
        flux[k][4] = (1.40e+00 * rsd[k][j][i][4] - 0.40e+00 * q) * u41;
      }
      
#pragma omp parallel for private (k,m)
      for (k = 1; k <= nz - 1 - 1; k += 1) {
        
#pragma omp parallel for private (m) firstprivate (tz2)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = frct[k][j][i][m] - tz2 * (flux[k + 1][m] - flux[k - 1][m]);
        }
      }
      
#pragma omp parallel for private (tmp,u21k,u31k,u41k,u51k,u21km1,u31km1,u41km1,u51km1,k)
      for (k = 1; k <= nz - 1; k += 1) {
        tmp = 1.0 / rsd[k][j][i][0];
        u21k = tmp * rsd[k][j][i][1];
        u31k = tmp * rsd[k][j][i][2];
        u41k = tmp * rsd[k][j][i][3];
        u51k = tmp * rsd[k][j][i][4];
        tmp = 1.0 / rsd[k - 1][j][i][0];
        u21km1 = tmp * rsd[k - 1][j][i][1];
        u31km1 = tmp * rsd[k - 1][j][i][2];
        u41km1 = tmp * rsd[k - 1][j][i][3];
        u51km1 = tmp * rsd[k - 1][j][i][4];
        flux[k][1] = tz3 * (u21k - u21km1);
        flux[k][2] = tz3 * (u31k - u31km1);
        flux[k][3] = 4.0 / 3.0 * tz3 * (u41k - u41km1);
        flux[k][4] = 0.50 * (1.0 - 1.40e+00 * 1.40e+00) * tz3 * (u21k * u21k + u31k * u31k + u41k * u41k - (u21km1 * u21km1 + u31km1 * u31km1 + u41km1 * u41km1)) + 1.0 / 6.0 * tz3 * (u41k * u41k - u41km1 * u41km1) + 1.40e+00 * 1.40e+00 * tz3 * (u51k - u51km1);
      }
      
#pragma omp parallel for private (k) firstprivate (tz1,tz3,dz1,dz2,dz3,dz4,dz5)
      for (k = 1; k <= nz - 1 - 1; k += 1) {
        frct[k][j][i][0] = frct[k][j][i][0] + dz1 * tz1 * (rsd[k + 1][j][i][0] - 2.0 * rsd[k][j][i][0] + rsd[k - 1][j][i][0]);
        frct[k][j][i][1] = frct[k][j][i][1] + tz3 * 1.00e-01 * 1.00e+00 * (flux[k + 1][1] - flux[k][1]) + dz2 * tz1 * (rsd[k + 1][j][i][1] - 2.0 * rsd[k][j][i][1] + rsd[k - 1][j][i][1]);
        frct[k][j][i][2] = frct[k][j][i][2] + tz3 * 1.00e-01 * 1.00e+00 * (flux[k + 1][2] - flux[k][2]) + dz3 * tz1 * (rsd[k + 1][j][i][2] - 2.0 * rsd[k][j][i][2] + rsd[k - 1][j][i][2]);
        frct[k][j][i][3] = frct[k][j][i][3] + tz3 * 1.00e-01 * 1.00e+00 * (flux[k + 1][3] - flux[k][3]) + dz4 * tz1 * (rsd[k + 1][j][i][3] - 2.0 * rsd[k][j][i][3] + rsd[k - 1][j][i][3]);
        frct[k][j][i][4] = frct[k][j][i][4] + tz3 * 1.00e-01 * 1.00e+00 * (flux[k + 1][4] - flux[k][4]) + dz5 * tz1 * (rsd[k + 1][j][i][4] - 2.0 * rsd[k][j][i][4] + rsd[k - 1][j][i][4]);
      }
//---------------------------------------------------------------------
// fourth-order dissipation
//---------------------------------------------------------------------
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 4; m += 1) {
        frct[1][j][i][m] = frct[1][j][i][m] - dssp * (+5.0 * rsd[1][j][i][m] - 4.0 * rsd[2][j][i][m] + rsd[3][j][i][m]);
        frct[2][j][i][m] = frct[2][j][i][m] - dssp * (- 4.0 * rsd[1][j][i][m] + 6.0 * rsd[2][j][i][m] - 4.0 * rsd[3][j][i][m] + rsd[4][j][i][m]);
      }
      
#pragma omp parallel for private (k,m)
      for (k = 3; k <= nz - 3 - 1; k += 1) {
        
#pragma omp parallel for private (m)
        for (m = 0; m <= 4; m += 1) {
          frct[k][j][i][m] = frct[k][j][i][m] - dssp * (rsd[k - 2][j][i][m] - 4.0 * rsd[k - 1][j][i][m] + 6.0 * rsd[k][j][i][m] - 4.0 * rsd[k + 1][j][i][m] + rsd[k + 2][j][i][m]);
        }
      }
      
#pragma omp parallel for private (m) firstprivate (dssp)
      for (m = 0; m <= 4; m += 1) {
        frct[nz - 3][j][i][m] = frct[nz - 3][j][i][m] - dssp * (rsd[nz - 5][j][i][m] - 4.0 * rsd[nz - 4][j][i][m] + 6.0 * rsd[nz - 3][j][i][m] - 4.0 * rsd[nz - 2][j][i][m]);
        frct[nz - 2][j][i][m] = frct[nz - 2][j][i][m] - dssp * (rsd[nz - 4][j][i][m] - 4.0 * rsd[nz - 3][j][i][m] + 5.0 * rsd[nz - 2][j][i][m]);
      }
    }
  }
}
