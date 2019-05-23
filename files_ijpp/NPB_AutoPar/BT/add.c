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
#include "timers.h"
//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
#include "omp.h" 

void add()
{
  int i;
  int j;
  int k;
  int m;
  if (timeron) 
    timer_start(11);
  
#pragma omp parallel for private (i,j,k,m)
  for (k = 1; k <= grid_points[2] - 2; k += 1) {
    
//#pragma omp parallel for private (i,j,m)
    for (j = 1; j <= grid_points[1] - 2; j += 1) {
      
//#pragma omp parallel for private (i,m)
      for (i = 1; i <= grid_points[0] - 2; i += 1) {
        
//#pragma omp parallel for private (m)
        for (m = 0; m <= 4; m += 1) {
          u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
        }
      }
    }
  }
  if (timeron) 
    timer_stop(11);
}
