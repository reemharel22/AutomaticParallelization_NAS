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
// set the initial values of independent variables based on tri-linear
// interpolation of boundary values in the computational space.
//
//---------------------------------------------------------------------
#include "omp.h" 

void setiv()
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
  double pxi;
  double peta;
  double pzeta;
  double ue_1jk[5];
  double ue_nx0jk[5];
  double ue_i1k[5];
  double ue_iny0k[5];
  double ue_ij1[5];
  double ue_ijnz[5];
  for (k = 1; k <= nz - 1 - 1; k += 1) {
    zeta = ((double )k) / (nz - 1);
    for (j = 1; j <= ny - 1 - 1; j += 1) {
      eta = ((double )j) / (ny0 - 1);
      for (i = 1; i <= nx - 1 - 1; i += 1) {
        xi = ((double )i) / (nx0 - 1);
        exact(0,j,k,ue_1jk);
        exact(nx0 - 1,j,k,ue_nx0jk);
        exact(i,0,k,ue_i1k);
        exact(i,ny0 - 1,k,ue_iny0k);
        exact(i,j,0,ue_ij1);
        exact(i,j,nz - 1,ue_ijnz);
        
#pragma omp parallel for private (pxi,peta,pzeta,m) firstprivate (xi,eta,zeta)
        for (m = 0; m <= 4; m += 1) {
          pxi = (1.0 - xi) * ue_1jk[m] + xi * ue_nx0jk[m];
          peta = (1.0 - eta) * ue_i1k[m] + eta * ue_iny0k[m];
          pzeta = (1.0 - zeta) * ue_ij1[m] + zeta * ue_ijnz[m];
          u[k][j][i][m] = pxi + peta + pzeta - pxi * peta - peta * pzeta - pzeta * pxi + pxi * peta * pzeta;
        }
      }
    }
  }
}
