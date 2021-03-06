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
// compute the right hand side based on exact solution
//---------------------------------------------------------------------
#include "omp.h" 

void exact_rhs()
{
  double dtemp[5];
  double xi;
  double eta;
  double zeta;
  double dtpp;
  int m;
  int i;
  int j;
  int k;
  int ip1;
  int im1;
  int jp1;
  int jm1;
  int km1;
  int kp1;
//---------------------------------------------------------------------
// initialize                                  
//---------------------------------------------------------------------
  for (k = 0; k <= grid_points[2] - 1; k += 1) {
    
#pragma omp parallel for private (m,i,j)
    for (j = 0; j <= grid_points[1] - 1; j += 1) {
      
#pragma omp parallel for private (m,i)
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        
#pragma omp parallel for private (m) firstprivate (k)
        for (m = 0; m <= 4; m += 1) {
          forcing[k][j][i][m] = 0.0;
        }
      }
    }
}
//---------------------------------------------------------------------
// xi-direction flux differences                      
//---------------------------------------------------------------------
    for (k = 1; k <= grid_points[2] - 2; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (j = 1; j <= grid_points[1] - 2; j += 1) {
        eta = ((double )j) * dnym1;
        for (i = 0; i <= grid_points[0] - 1; i += 1) {
          xi = ((double )i) * dnxm1;
          exact_solution(xi,eta,zeta,dtemp);
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            ue[i][m] = dtemp[m];
          }
          dtpp = 1.0 / dtemp[0];
          
#pragma omp parallel for private (m) firstprivate (dtpp)
          for (m = 1; m <= 4; m += 1) {
            buf[i][m] = dtpp * dtemp[m];
          }
          cuf[i] = buf[i][1] * buf[i][1];
          buf[i][0] = cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3];
          q[i] = 0.5 * (buf[i][1] * ue[i][1] + buf[i][2] * ue[i][2] + buf[i][3] * ue[i][3]);
        }
        
#pragma omp parallel for private (ip1,im1,i) firstprivate (tx2,xxcon1,xxcon2,xxcon3,xxcon4,xxcon5,dx1tx1,dx2tx1,dx3tx1,dx4tx1,dx5tx1,c1,c2)
        for (i = 1; i <= grid_points[0] - 2; i += 1) {
          im1 = i - 1;
          ip1 = i + 1;
          forcing[k][j][i][0] = forcing[k][j][i][0] - tx2 * (ue[ip1][1] - ue[im1][1]) + dx1tx1 * (ue[ip1][0] - 2.0 * ue[i][0] + ue[im1][0]);
          forcing[k][j][i][1] = forcing[k][j][i][1] - tx2 * (ue[ip1][1] * buf[ip1][1] + c2 * (ue[ip1][4] - q[ip1]) - (ue[im1][1] * buf[im1][1] + c2 * (ue[im1][4] - q[im1]))) + xxcon1 * (buf[ip1][1] - 2.0 * buf[i][1] + buf[im1][1]) + dx2tx1 * (ue[ip1][1] - 2.0 * ue[i][1] + ue[im1][1]);
          forcing[k][j][i][2] = forcing[k][j][i][2] - tx2 * (ue[ip1][2] * buf[ip1][1] - ue[im1][2] * buf[im1][1]) + xxcon2 * (buf[ip1][2] - 2.0 * buf[i][2] + buf[im1][2]) + dx3tx1 * (ue[ip1][2] - 2.0 * ue[i][2] + ue[im1][2]);
          forcing[k][j][i][3] = forcing[k][j][i][3] - tx2 * (ue[ip1][3] * buf[ip1][1] - ue[im1][3] * buf[im1][1]) + xxcon2 * (buf[ip1][3] - 2.0 * buf[i][3] + buf[im1][3]) + dx4tx1 * (ue[ip1][3] - 2.0 * ue[i][3] + ue[im1][3]);
          forcing[k][j][i][4] = forcing[k][j][i][4] - tx2 * (buf[ip1][1] * (c1 * ue[ip1][4] - c2 * q[ip1]) - buf[im1][1] * (c1 * ue[im1][4] - c2 * q[im1])) + 0.5 * xxcon3 * (buf[ip1][0] - 2.0 * buf[i][0] + buf[im1][0]) + xxcon4 * (cuf[ip1] - 2.0 * cuf[i] + cuf[im1]) + xxcon5 * (buf[ip1][4] - 2.0 * buf[i][4] + buf[im1][4]) + dx5tx1 * (ue[ip1][4] - 2.0 * ue[i][4] + ue[im1][4]);
        }
//---------------------------------------------------------------------
// Fourth-order dissipation                         
//---------------------------------------------------------------------
        
#pragma omp parallel for private (i,m)
        for (m = 0; m <= 4; m += 1) {
          i = 1;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (5.0 * ue[i][m] - 4.0 * ue[i + 1][m] + ue[i + 2][m]);
          i = 2;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (- 4.0 * ue[i - 1][m] + 6.0 * ue[i][m] - 4.0 * ue[i + 1][m] + ue[i + 2][m]);
        }
        
#pragma omp parallel for private (m,i)
        for (i = 3; i <= grid_points[0] - 4; i += 1) {
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[i - 2][m] - 4.0 * ue[i - 1][m] + 6.0 * ue[i][m] - 4.0 * ue[i + 1][m] + ue[i + 2][m]);
          }
        }
        
#pragma omp parallel for private (i,m) firstprivate (dssp)
        for (m = 0; m <= 4; m += 1) {
          i = grid_points[0] - 3;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[i - 2][m] - 4.0 * ue[i - 1][m] + 6.0 * ue[i][m] - 4.0 * ue[i + 1][m]);
          i = grid_points[0] - 2;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[i - 2][m] - 4.0 * ue[i - 1][m] + 5.0 * ue[i][m]);
        }
      }
    }
//---------------------------------------------------------------------
// eta-direction flux differences             
//---------------------------------------------------------------------
    for (k = 1; k <= grid_points[2] - 2; k += 1) {
      zeta = ((double )k) * dnzm1;
      for (i = 1; i <= grid_points[0] - 2; i += 1) {
        xi = ((double )i) * dnxm1;
        for (j = 0; j <= grid_points[1] - 1; j += 1) {
          eta = ((double )j) * dnym1;
          exact_solution(xi,eta,zeta,dtemp);
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            ue[j][m] = dtemp[m];
          }
          dtpp = 1.0 / dtemp[0];
          
#pragma omp parallel for private (m) firstprivate (dtpp)
          for (m = 1; m <= 4; m += 1) {
            buf[j][m] = dtpp * dtemp[m];
          }
          cuf[j] = buf[j][2] * buf[j][2];
          buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3];
          q[j] = 0.5 * (buf[j][1] * ue[j][1] + buf[j][2] * ue[j][2] + buf[j][3] * ue[j][3]);
        }
        
#pragma omp parallel for private (jp1,jm1,j) firstprivate (ty2,yycon1,yycon2,yycon3,yycon4,yycon5,dy1ty1,dy2ty1,dy3ty1,dy4ty1,dy5ty1,c1,c2)
        for (j = 1; j <= grid_points[1] - 2; j += 1) {
          jm1 = j - 1;
          jp1 = j + 1;
          forcing[k][j][i][0] = forcing[k][j][i][0] - ty2 * (ue[jp1][2] - ue[jm1][2]) + dy1ty1 * (ue[jp1][0] - 2.0 * ue[j][0] + ue[jm1][0]);
          forcing[k][j][i][1] = forcing[k][j][i][1] - ty2 * (ue[jp1][1] * buf[jp1][2] - ue[jm1][1] * buf[jm1][2]) + yycon2 * (buf[jp1][1] - 2.0 * buf[j][1] + buf[jm1][1]) + dy2ty1 * (ue[jp1][1] - 2.0 * ue[j][1] + ue[jm1][1]);
          forcing[k][j][i][2] = forcing[k][j][i][2] - ty2 * (ue[jp1][2] * buf[jp1][2] + c2 * (ue[jp1][4] - q[jp1]) - (ue[jm1][2] * buf[jm1][2] + c2 * (ue[jm1][4] - q[jm1]))) + yycon1 * (buf[jp1][2] - 2.0 * buf[j][2] + buf[jm1][2]) + dy3ty1 * (ue[jp1][2] - 2.0 * ue[j][2] + ue[jm1][2]);
          forcing[k][j][i][3] = forcing[k][j][i][3] - ty2 * (ue[jp1][3] * buf[jp1][2] - ue[jm1][3] * buf[jm1][2]) + yycon2 * (buf[jp1][3] - 2.0 * buf[j][3] + buf[jm1][3]) + dy4ty1 * (ue[jp1][3] - 2.0 * ue[j][3] + ue[jm1][3]);
          forcing[k][j][i][4] = forcing[k][j][i][4] - ty2 * (buf[jp1][2] * (c1 * ue[jp1][4] - c2 * q[jp1]) - buf[jm1][2] * (c1 * ue[jm1][4] - c2 * q[jm1])) + 0.5 * yycon3 * (buf[jp1][0] - 2.0 * buf[j][0] + buf[jm1][0]) + yycon4 * (cuf[jp1] - 2.0 * cuf[j] + cuf[jm1]) + yycon5 * (buf[jp1][4] - 2.0 * buf[j][4] + buf[jm1][4]) + dy5ty1 * (ue[jp1][4] - 2.0 * ue[j][4] + ue[jm1][4]);
        }
//---------------------------------------------------------------------
// Fourth-order dissipation                      
//---------------------------------------------------------------------
        
#pragma omp parallel for private (j,m)
        for (m = 0; m <= 4; m += 1) {
          j = 1;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (5.0 * ue[j][m] - 4.0 * ue[j + 1][m] + ue[j + 2][m]);
          j = 2;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (- 4.0 * ue[j - 1][m] + 6.0 * ue[j][m] - 4.0 * ue[j + 1][m] + ue[j + 2][m]);
        }
        
#pragma omp parallel for private (m,j)
        for (j = 3; j <= grid_points[1] - 4; j += 1) {
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[j - 2][m] - 4.0 * ue[j - 1][m] + 6.0 * ue[j][m] - 4.0 * ue[j + 1][m] + ue[j + 2][m]);
          }
        }
        
#pragma omp parallel for private (j,m) firstprivate (dssp)
        for (m = 0; m <= 4; m += 1) {
          j = grid_points[1] - 3;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[j - 2][m] - 4.0 * ue[j - 1][m] + 6.0 * ue[j][m] - 4.0 * ue[j + 1][m]);
          j = grid_points[1] - 2;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[j - 2][m] - 4.0 * ue[j - 1][m] + 5.0 * ue[j][m]);
        }
      }
    }
//---------------------------------------------------------------------
// zeta-direction flux differences                      
//---------------------------------------------------------------------
    for (j = 1; j <= grid_points[1] - 2; j += 1) {
      eta = ((double )j) * dnym1;
      for (i = 1; i <= grid_points[0] - 2; i += 1) {
        xi = ((double )i) * dnxm1;
        for (k = 0; k <= grid_points[2] - 1; k += 1) {
          zeta = ((double )k) * dnzm1;
          exact_solution(xi,eta,zeta,dtemp);
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            ue[k][m] = dtemp[m];
          }
          dtpp = 1.0 / dtemp[0];
          
#pragma omp parallel for private (m) firstprivate (dtpp)
          for (m = 1; m <= 4; m += 1) {
            buf[k][m] = dtpp * dtemp[m];
          }
          cuf[k] = buf[k][3] * buf[k][3];
          buf[k][0] = cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2];
          q[k] = 0.5 * (buf[k][1] * ue[k][1] + buf[k][2] * ue[k][2] + buf[k][3] * ue[k][3]);
        }
        
#pragma omp parallel for private (km1,kp1,k) firstprivate (tz2,zzcon1,zzcon2,zzcon3,zzcon4,zzcon5,dz1tz1,dz2tz1,dz3tz1,dz4tz1,dz5tz1,c1,c2)
        for (k = 1; k <= grid_points[2] - 2; k += 1) {
          km1 = k - 1;
          kp1 = k + 1;
          forcing[k][j][i][0] = forcing[k][j][i][0] - tz2 * (ue[kp1][3] - ue[km1][3]) + dz1tz1 * (ue[kp1][0] - 2.0 * ue[k][0] + ue[km1][0]);
          forcing[k][j][i][1] = forcing[k][j][i][1] - tz2 * (ue[kp1][1] * buf[kp1][3] - ue[km1][1] * buf[km1][3]) + zzcon2 * (buf[kp1][1] - 2.0 * buf[k][1] + buf[km1][1]) + dz2tz1 * (ue[kp1][1] - 2.0 * ue[k][1] + ue[km1][1]);
          forcing[k][j][i][2] = forcing[k][j][i][2] - tz2 * (ue[kp1][2] * buf[kp1][3] - ue[km1][2] * buf[km1][3]) + zzcon2 * (buf[kp1][2] - 2.0 * buf[k][2] + buf[km1][2]) + dz3tz1 * (ue[kp1][2] - 2.0 * ue[k][2] + ue[km1][2]);
          forcing[k][j][i][3] = forcing[k][j][i][3] - tz2 * (ue[kp1][3] * buf[kp1][3] + c2 * (ue[kp1][4] - q[kp1]) - (ue[km1][3] * buf[km1][3] + c2 * (ue[km1][4] - q[km1]))) + zzcon1 * (buf[kp1][3] - 2.0 * buf[k][3] + buf[km1][3]) + dz4tz1 * (ue[kp1][3] - 2.0 * ue[k][3] + ue[km1][3]);
          forcing[k][j][i][4] = forcing[k][j][i][4] - tz2 * (buf[kp1][3] * (c1 * ue[kp1][4] - c2 * q[kp1]) - buf[km1][3] * (c1 * ue[km1][4] - c2 * q[km1])) + 0.5 * zzcon3 * (buf[kp1][0] - 2.0 * buf[k][0] + buf[km1][0]) + zzcon4 * (cuf[kp1] - 2.0 * cuf[k] + cuf[km1]) + zzcon5 * (buf[kp1][4] - 2.0 * buf[k][4] + buf[km1][4]) + dz5tz1 * (ue[kp1][4] - 2.0 * ue[k][4] + ue[km1][4]);
        }
//---------------------------------------------------------------------
// Fourth-order dissipation                        
//---------------------------------------------------------------------
        
#pragma omp parallel for private (k,m)
        for (m = 0; m <= 4; m += 1) {
          k = 1;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (5.0 * ue[k][m] - 4.0 * ue[k + 1][m] + ue[k + 2][m]);
          k = 2;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (- 4.0 * ue[k - 1][m] + 6.0 * ue[k][m] - 4.0 * ue[k + 1][m] + ue[k + 2][m]);
        }
        
#pragma omp parallel for private (m,k)
        for (k = 3; k <= grid_points[2] - 4; k += 1) {
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[k - 2][m] - 4.0 * ue[k - 1][m] + 6.0 * ue[k][m] - 4.0 * ue[k + 1][m] + ue[k + 2][m]);
          }
        }
        
#pragma omp parallel for private (k,m) firstprivate (dssp)
        for (m = 0; m <= 4; m += 1) {
          k = grid_points[2] - 3;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[k - 2][m] - 4.0 * ue[k - 1][m] + 6.0 * ue[k][m] - 4.0 * ue[k + 1][m]);
          k = grid_points[2] - 2;
          forcing[k][j][i][m] = forcing[k][j][i][m] - dssp * (ue[k - 2][m] - 4.0 * ue[k - 1][m] + 5.0 * ue[k][m]);
        }
      }
    }
//---------------------------------------------------------------------
// now change the sign of the forcing function, 
//---------------------------------------------------------------------
    
#pragma omp parallel for private (m,i,j,k)
    for (k = 1; k <= grid_points[2] - 2; k += 1) {
      
#pragma omp parallel for private (m,i,j)
      for (j = 1; j <= grid_points[1] - 2; j += 1) {
        
#pragma omp parallel for private (m,i)
        for (i = 1; i <= grid_points[0] - 2; i += 1) {
          
#pragma omp parallel for private (m)
          for (m = 0; m <= 4; m += 1) {
            forcing[k][j][i][m] = - 1.0 * forcing[k][j][i][m];
          }
        }
      }
    }
//end parallel
  
}
