//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB SP code. This OpenMP  //
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
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
#define N 100
#define PROBLEM_SIZE 10
#include <omp.h> 

void x_solve()
{
  int i;
  int j;
  int k;
  int i1;
  int i2;
  int m;
  int IMAXP;
  int grid_points[100];
  double ru1;
  double fac1;
  double fac2;
  double comz6;
  double speed[100][100][100];
  double us[100][100][100];
  double comz1;
  double comz2;
  double comz3;
  double comz4;
  double comz5;
  double c1c5;
  double dxmax;
  double dttx1;
  double dttx2;
  double dx1;
  double dx2;
  double dx3;
  double dx4;
  double dx5;
  double dx6;
  double bt;
  double c2c4;
  double c2c3;
  double c3c4;
  double con43;
  double c2dttx1;
  double rho_i[100][100][100];
  double rhs[100][100][100][100];
//if (timeron) timer_start(t_xsolve);
  int nz2;
  int ny2;
  int nx2;
  for (k = 1; k <= nz2; k += 1) {
    double lhs[IMAXP + 1][IMAXP + 1][5];
//double rhon[PROBLEM_SIZE];
//double cv  [PROBLEM_SIZE];
    double lhsp[IMAXP + 1][IMAXP + 1][5];
    double lhsm[IMAXP + 1][IMAXP + 1][5];
//lhsinit(nx2+1, ny2);
/* lhsinit - START */
    int j_lhsinit;
    int m_lhsinit;
//---------------------------------------------------------------------
// zap the whole left hand side for starters
// set all diagonal values to 1. This is overkill, but convenient
//---------------------------------------------------------------------
    
#pragma omp parallel for private (j_lhsinit,m_lhsinit)
    for (j_lhsinit = 1; j_lhsinit <= ny2; j_lhsinit += 1) {
      
#pragma omp parallel for private (m_lhsinit)
      for (m_lhsinit = 0; m_lhsinit <= 4; m_lhsinit += 1) {
        lhs[j_lhsinit][0][m_lhsinit] = 0.0;
        lhsp[j_lhsinit][0][m_lhsinit] = 0.0;
        lhsm[j_lhsinit][0][m_lhsinit] = 0.0;
        lhs[j_lhsinit][nx2 + 1][m_lhsinit] = 0.0;
        lhsp[j_lhsinit][nx2 + 1][m_lhsinit] = 0.0;
        lhsm[j_lhsinit][nx2 + 1][m_lhsinit] = 0.0;
      }
      lhs[j_lhsinit][0][2] = 1.0;
      lhsp[j_lhsinit][0][2] = 1.0;
      lhsm[j_lhsinit][0][2] = 1.0;
      lhs[j_lhsinit][nx2 + 1][2] = 1.0;
      lhsp[j_lhsinit][nx2 + 1][2] = 1.0;
      lhsm[j_lhsinit][nx2 + 1][2] = 1.0;
    }
/* lhsinit - END */
//---------------------------------------------------------------------
// Computes the left hand side for the three x-factors  
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// first fill the lhs for the u-eigenvalue                   
//---------------------------------------------------------------------
    for (j = 1; j <= ny2; j += 1) {
      double cv[162];
      double rhon[10];
      
#pragma omp parallel for private (ru1,i) firstprivate (c3c4)
      for (i = 0; i <= grid_points[0] - 1; i += 1) {
        ru1 = c3c4 * rho_i[k][j][i];
        cv[i] = us[k][j][i];
//rhon[i] = max(max(dx2+con43*ru1,dx5+c1c5*ru1), max(dxmax+ru1,dx1));
      }
      
#pragma omp parallel for private (i) firstprivate (dttx1,dttx2,c2dttx1)
      for (i = 1; i <= nx2; i += 1) {
        lhs[j][i][0] = 0.0;
        lhs[j][i][1] = -dttx2 * cv[i - 1] - dttx1 * rhon[i - 1];
        lhs[j][i][2] = 1.0 + c2dttx1 * rhon[i];
        lhs[j][i][3] = dttx2 * cv[i + 1] - dttx1 * rhon[i + 1];
        lhs[j][i][4] = 0.0;
      }
    }
//---------------------------------------------------------------------
// add fourth order dissipation                             
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i,j)
    for (j = 1; j <= ny2; j += 1) {
      i = 1;
      lhs[j][i][2] = lhs[j][i][2] + comz5;
      lhs[j][i][3] = lhs[j][i][3] - comz4;
      lhs[j][i][4] = lhs[j][i][4] + comz1;
      lhs[j][i + 1][1] = lhs[j][i + 1][1] - comz4;
      lhs[j][i + 1][2] = lhs[j][i + 1][2] + comz6;
      lhs[j][i + 1][3] = lhs[j][i + 1][3] - comz4;
      lhs[j][i + 1][4] = lhs[j][i + 1][4] + comz1;
    }
    
#pragma omp parallel for private (i,j)
    for (j = 1; j <= ny2; j += 1) {
      
#pragma omp parallel for private (i)
      for (i = 3; i <= grid_points[0] - 4; i += 1) {
        lhs[j][i][0] = lhs[j][i][0] + comz1;
        lhs[j][i][1] = lhs[j][i][1] - comz4;
        lhs[j][i][2] = lhs[j][i][2] + comz6;
        lhs[j][i][3] = lhs[j][i][3] - comz4;
        lhs[j][i][4] = lhs[j][i][4] + comz1;
      }
    }
    
#pragma omp parallel for private (i,j) firstprivate (comz6,comz1,comz4,comz5)
    for (j = 1; j <= ny2; j += 1) {
      i = grid_points[0] - 3;
      lhs[j][i][0] = lhs[j][i][0] + comz1;
      lhs[j][i][1] = lhs[j][i][1] - comz4;
      lhs[j][i][2] = lhs[j][i][2] + comz6;
      lhs[j][i][3] = lhs[j][i][3] - comz4;
      lhs[j][i + 1][0] = lhs[j][i + 1][0] + comz1;
      lhs[j][i + 1][1] = lhs[j][i + 1][1] - comz4;
      lhs[j][i + 1][2] = lhs[j][i + 1][2] + comz5;
    }
//---------------------------------------------------------------------
// subsequently, fill the other factors (u+c), (u-c) by adding to 
// the first  
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i,j) firstprivate (nx2)
    for (j = 1; j <= ny2; j += 1) {
      
#pragma omp parallel for private (i) firstprivate (dttx2)
      for (i = 1; i <= nx2; i += 1) {
        lhsp[j][i][0] = lhs[j][i][0];
        lhsp[j][i][1] = lhs[j][i][1] - dttx2 * speed[k][j][i - 1];
        lhsp[j][i][2] = lhs[j][i][2];
        lhsp[j][i][3] = lhs[j][i][3] + dttx2 * speed[k][j][i + 1];
        lhsp[j][i][4] = lhs[j][i][4];
        lhsm[j][i][0] = lhs[j][i][0];
        lhsm[j][i][1] = lhs[j][i][1] + dttx2 * speed[k][j][i - 1];
        lhsm[j][i][2] = lhs[j][i][2];
        lhsm[j][i][3] = lhs[j][i][3] - dttx2 * speed[k][j][i + 1];
        lhsm[j][i][4] = lhs[j][i][4];
      }
    }
//---------------------------------------------------------------------
// FORWARD ELIMINATION  
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// perform the Thomas algorithm; first, FORWARD ELIMINATION     
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i1,i2,fac1,i,j,m)
    for (j = 1; j <= ny2; j += 1) {
      for (i = 0; i <= grid_points[0] - 3; i += 1) {
        i1 = i + 1;
        i2 = i + 2;
        fac1 = 1.0 / lhs[j][i][2];
        lhs[j][i][3] = fac1 * lhs[j][i][3];
        lhs[j][i][4] = fac1 * lhs[j][i][4];
        
#pragma omp parallel for private (m) firstprivate (fac1)
        for (m = 0; m <= 2; m += 1) {
          rhs[k][j][i][m] = fac1 * rhs[k][j][i][m];
        }
        lhs[j][i1][2] = lhs[j][i1][2] - lhs[j][i1][1] * lhs[j][i][3];
        lhs[j][i1][3] = lhs[j][i1][3] - lhs[j][i1][1] * lhs[j][i][4];
        
#pragma omp parallel for private (m) firstprivate (i1)
        for (m = 0; m <= 2; m += 1) {
          rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs[j][i1][1] * rhs[k][j][i][m];
        }
        lhs[j][i2][1] = lhs[j][i2][1] - lhs[j][i2][0] * lhs[j][i][3];
        lhs[j][i2][2] = lhs[j][i2][2] - lhs[j][i2][0] * lhs[j][i][4];
        
#pragma omp parallel for private (m) firstprivate (i2)
        for (m = 0; m <= 2; m += 1) {
          rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs[j][i2][0] * rhs[k][j][i][m];
        }
      }
    }
//---------------------------------------------------------------------
// The last two rows in this grid block are a bit different, 
// since they for (not have two more rows available for the
// elimination of off-diagonal entries
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i,i1,fac1,fac2,j,m)
    for (j = 1; j <= ny2; j += 1) {
      i = grid_points[0] - 2;
      i1 = grid_points[0] - 1;
      fac1 = 1.0 / lhs[j][i][2];
      lhs[j][i][3] = fac1 * lhs[j][i][3];
      lhs[j][i][4] = fac1 * lhs[j][i][4];
      
#pragma omp parallel for private (m) firstprivate (fac1)
      for (m = 0; m <= 2; m += 1) {
        rhs[k][j][i][m] = fac1 * rhs[k][j][i][m];
      }
      lhs[j][i1][2] = lhs[j][i1][2] - lhs[j][i1][1] * lhs[j][i][3];
      lhs[j][i1][3] = lhs[j][i1][3] - lhs[j][i1][1] * lhs[j][i][4];
      
#pragma omp parallel for private (m) firstprivate (i)
      for (m = 0; m <= 2; m += 1) {
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs[j][i1][1] * rhs[k][j][i][m];
      }
//---------------------------------------------------------------------
// scale the last row immediately 
//---------------------------------------------------------------------
      fac2 = 1.0 / lhs[j][i1][2];
      
#pragma omp parallel for private (m) firstprivate (i1,fac2)
      for (m = 0; m <= 2; m += 1) {
        rhs[k][j][i1][m] = fac2 * rhs[k][j][i1][m];
      }
    }
//---------------------------------------------------------------------
// for (the u+c and the u-c factors                 
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i1,i2,m,fac1,i,j)
    for (j = 1; j <= ny2; j += 1) {
      for (i = 0; i <= grid_points[0] - 3; i += 1) {
        i1 = i + 1;
        i2 = i + 2;
        m = 3;
        fac1 = 1.0 / lhsp[j][i][2];
        lhsp[j][i][3] = fac1 * lhsp[j][i][3];
        lhsp[j][i][4] = fac1 * lhsp[j][i][4];
        rhs[k][j][i][m] = fac1 * rhs[k][j][i][m];
        lhsp[j][i1][2] = lhsp[j][i1][2] - lhsp[j][i1][1] * lhsp[j][i][3];
        lhsp[j][i1][3] = lhsp[j][i1][3] - lhsp[j][i1][1] * lhsp[j][i][4];
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp[j][i1][1] * rhs[k][j][i][m];
        lhsp[j][i2][1] = lhsp[j][i2][1] - lhsp[j][i2][0] * lhsp[j][i][3];
        lhsp[j][i2][2] = lhsp[j][i2][2] - lhsp[j][i2][0] * lhsp[j][i][4];
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp[j][i2][0] * rhs[k][j][i][m];
        m = 4;
        fac1 = 1.0 / lhsm[j][i][2];
        lhsm[j][i][3] = fac1 * lhsm[j][i][3];
        lhsm[j][i][4] = fac1 * lhsm[j][i][4];
        rhs[k][j][i][m] = fac1 * rhs[k][j][i][m];
        lhsm[j][i1][2] = lhsm[j][i1][2] - lhsm[j][i1][1] * lhsm[j][i][3];
        lhsm[j][i1][3] = lhsm[j][i1][3] - lhsm[j][i1][1] * lhsm[j][i][4];
        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm[j][i1][1] * rhs[k][j][i][m];
        lhsm[j][i2][1] = lhsm[j][i2][1] - lhsm[j][i2][0] * lhsm[j][i][3];
        lhsm[j][i2][2] = lhsm[j][i2][2] - lhsm[j][i2][0] * lhsm[j][i][4];
        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm[j][i2][0] * rhs[k][j][i][m];
      }
    }
//---------------------------------------------------------------------
// And again the last two rows separately
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i,i1,m,fac1,j)
    for (j = 1; j <= ny2; j += 1) {
      i = grid_points[0] - 2;
      i1 = grid_points[0] - 1;
      m = 3;
      fac1 = 1.0 / lhsp[j][i][2];
      lhsp[j][i][3] = fac1 * lhsp[j][i][3];
      lhsp[j][i][4] = fac1 * lhsp[j][i][4];
      rhs[k][j][i][m] = fac1 * rhs[k][j][i][m];
      lhsp[j][i1][2] = lhsp[j][i1][2] - lhsp[j][i1][1] * lhsp[j][i][3];
      lhsp[j][i1][3] = lhsp[j][i1][3] - lhsp[j][i1][1] * lhsp[j][i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp[j][i1][1] * rhs[k][j][i][m];
      m = 4;
      fac1 = 1.0 / lhsm[j][i][2];
      lhsm[j][i][3] = fac1 * lhsm[j][i][3];
      lhsm[j][i][4] = fac1 * lhsm[j][i][4];
      rhs[k][j][i][m] = fac1 * rhs[k][j][i][m];
      lhsm[j][i1][2] = lhsm[j][i1][2] - lhsm[j][i1][1] * lhsm[j][i][3];
      lhsm[j][i1][3] = lhsm[j][i1][3] - lhsm[j][i1][1] * lhsm[j][i][4];
      rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm[j][i1][1] * rhs[k][j][i][m];
//---------------------------------------------------------------------
// Scale the last row immediately
//---------------------------------------------------------------------
      rhs[k][j][i1][3] = rhs[k][j][i1][3] / lhsp[j][i1][2];
      rhs[k][j][i1][4] = rhs[k][j][i1][4] / lhsm[j][i1][2];
    }
//---------------------------------------------------------------------
// BACKSUBSTITUTION 
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i,i1,j,m)
    for (j = 1; j <= ny2; j += 1) {
      i = grid_points[0] - 2;
      i1 = grid_points[0] - 1;
      
#pragma omp parallel for private (m)
      for (m = 0; m <= 2; m += 1) {
        rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][i][3] * rhs[k][j][i1][m];
      }
      rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[j][i][3] * rhs[k][j][i1][3];
      rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[j][i][3] * rhs[k][j][i1][4];
    }
//---------------------------------------------------------------------
// The first three factors
//---------------------------------------------------------------------
    
#pragma omp parallel for private (i1,i2,i,j,m)
    for (j = 1; j <= ny2; j += 1) {
      for (i = grid_points[0] - 3; i >= 0; i += -1) {
        i1 = i + 1;
        i2 = i + 2;
        
#pragma omp parallel for private (m)
        for (m = 0; m <= 2; m += 1) {
          rhs[k][j][i][m] = rhs[k][j][i][m] - lhs[j][i][3] * rhs[k][j][i1][m] - lhs[j][i][4] * rhs[k][j][i2][m];
        }
//-------------------------------------------------------------------
// And the remaining two
//-------------------------------------------------------------------
        rhs[k][j][i][3] = rhs[k][j][i][3] - lhsp[j][i][3] * rhs[k][j][i1][3] - lhsp[j][i][4] * rhs[k][j][i2][3];
        rhs[k][j][i][4] = rhs[k][j][i][4] - lhsm[j][i][3] * rhs[k][j][i1][4] - lhsm[j][i][4] * rhs[k][j][i2][4];
      }
    }
  }
//---------------------------------------------------------------------
// Do the block-diagonal inversion          
//---------------------------------------------------------------------
//ninvr();
/* ninvr - START */
  int i_ninvr;
  int j_ninvr;
  int k_ninvr;
  double r1_ninvr;
  double r2_ninvr;
  double r3_ninvr;
  double r4_ninvr;
  double r5_ninvr;
  double t1_ninvr;
  double t2_ninvr;
  
#pragma omp parallel for private (r1_ninvr,r2_ninvr,r3_ninvr,r4_ninvr,r5_ninvr,t1_ninvr,t2_ninvr,i_ninvr,j_ninvr,k_ninvr) firstprivate (nz2,ny2)
  for (k_ninvr = 1; k_ninvr <= nz2; k_ninvr += 1) {
    
#pragma omp parallel for private (r1_ninvr,r2_ninvr,r3_ninvr,r4_ninvr,r5_ninvr,t1_ninvr,t2_ninvr,i_ninvr,j_ninvr) firstprivate (nx2)
    for (j_ninvr = 1; j_ninvr <= ny2; j_ninvr += 1) {
      
#pragma omp parallel for private (r1_ninvr,r2_ninvr,r3_ninvr,r4_ninvr,r5_ninvr,t1_ninvr,t2_ninvr,i_ninvr) firstprivate (bt)
      for (i_ninvr = 1; i_ninvr <= nx2; i_ninvr += 1) {
        r1_ninvr = rhs[k_ninvr][j_ninvr][i_ninvr][0];
        r2_ninvr = rhs[k_ninvr][j_ninvr][i_ninvr][1];
        r3_ninvr = rhs[k_ninvr][j_ninvr][i_ninvr][2];
        r4_ninvr = rhs[k_ninvr][j_ninvr][i_ninvr][3];
        r5_ninvr = rhs[k_ninvr][j_ninvr][i_ninvr][4];
        t1_ninvr = bt * r3_ninvr;
        t2_ninvr = 0.5 * (r4_ninvr + r5_ninvr);
        rhs[k_ninvr][j_ninvr][i_ninvr][0] = -r2_ninvr;
        rhs[k_ninvr][j_ninvr][i_ninvr][1] = r1_ninvr;
        rhs[k_ninvr][j_ninvr][i_ninvr][2] = bt * (r4_ninvr - r5_ninvr);
        rhs[k_ninvr][j_ninvr][i_ninvr][3] = -t1_ninvr + t2_ninvr;
        rhs[k_ninvr][j_ninvr][i_ninvr][4] = t1_ninvr + t2_ninvr;
      }
    }
  }
/* ninvr - END */
}
