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
#include "header.h"
//------------------------------------------------------------------
// initialize double precision array a with length of n
//------------------------------------------------------------------
#include "omp.h" 

void reciprocal(double a[],int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = 1.0 / a[i];
  }
}
//------------------------------------------------------------------
// initialize double precision array a with length of n
//------------------------------------------------------------------

void r_init_omp(double a[],int n,double _const)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n,_const)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = _const;
  }
}
//------------------------------------------------------------------
// initialize double precision array a with length of n
//------------------------------------------------------------------

void r_init(double a[],int n,double _const)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n,_const)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = _const;
  }
}
//------------------------------------------------------------------
// initialize integer array a with length of n
//------------------------------------------------------------------

void nr_init_omp(int a[],int n,int _const)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n,_const)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = _const;
  }
}
//------------------------------------------------------------------
// initialize integer array a with length of n
//------------------------------------------------------------------

void nr_init(int a[],int n,int _const)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n,_const)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = _const;
  }
}
//------------------------------------------------------------------
// initialize logical array a with length of n
//------------------------------------------------------------------

void l_init_omp(logical a[],int n,logical _const)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = _const;
  }
}
//------------------------------------------------------------------
// initialize logical array a with length of n
//------------------------------------------------------------------

void l_init(logical a[],int n,logical _const)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = _const;
  }
}
//------------------------------------------------------------------
// copy array of integers b to a, the length of array is n
//------------------------------------------------------------------

void ncopy(int a[],int b[],int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = b[i];
  }
}
//------------------------------------------------------------------
// copy double precision array b to a, the length of array is n
//------------------------------------------------------------------

void copy(double a[],double b[],int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = b[i];
  }
}
//-----------------------------------------------------------------
// a=b*c1
//-----------------------------------------------------------------

void adds2m1(double a[],double b[],double c1,int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (c1,n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = a[i] + c1 * b[i];
  }
}
//-----------------------------------------------------------------
// a=c1*a+b
//-----------------------------------------------------------------

void adds1m1(double a[],double b[],double c1,int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (c1,n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = c1 * a[i] + b[i];
  }
}
//------------------------------------------------------------------
// a=a*b
//------------------------------------------------------------------

void col2(double a[],double b[],int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = a[i] * b[i];
  }
}
//------------------------------------------------------------------
// zero out array of integers 
//------------------------------------------------------------------

void nrzero(int na[],int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    na[i] = 0;
  }
}
//------------------------------------------------------------------
// a=a+b
//------------------------------------------------------------------

void add2(double a[],double b[],int n)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = a[i] + b[i];
  }
}
//------------------------------------------------------------------
// calculate the integral of ta1 over the whole domain
//------------------------------------------------------------------

double calc_norm()
{
  double total;
  double ieltotal;
  int iel;
  int k;
  int j;
  int i;
  int isize;
  total = 0.0;
  
#pragma omp parallel for private (ieltotal,isize,iel,k,j,i) reduction (+:total) firstprivate (nelt)
  for (iel = 0; iel <= nelt - 1; iel += 1) {
    ieltotal = 0.0;
    isize = size_e[iel];
    
#pragma omp parallel for private (k,j,i) reduction (+:ieltotal)
    for (k = 0; k <= 4; k += 1) {
      
#pragma omp parallel for private (j,i) reduction (+:ieltotal)
      for (j = 0; j <= 4; j += 1) {
        
#pragma omp parallel for private (i) reduction (+:ieltotal) firstprivate (isize)
        for (i = 0; i <= 4; i += 1) {
          ieltotal = ieltotal + ta1[iel][k][j][i] * w3m1[k][j][i] * jacm1_s[isize][k][j][i];
        }
      }
    }
    total = total + ieltotal;
  }
  return total;
}
//-----------------------------------------------------------------
// input array frontier, perform (potentially) parallel add so that
// the output frontier[i] has sum of frontier[1]+frontier[2]+...+frontier[i]
//-----------------------------------------------------------------

void parallel_add(int frontier[])
{
  int nellog;
  int i;
  int ahead;
  int ii;
  int ntemp;
  int n1;
  int ntemp1;
  int iel;
  nellog = 0;
  iel = 1;
  do {
    iel = iel * 2;
    nellog = nellog + 1;
  }while (iel < nelt);
  ntemp = 1;
  for (i = 0; i <= nellog - 1; i += 1) {
    n1 = ntemp * 2;
    for (iel = n1; iel <= nelt; iel += n1) {
      ahead = frontier[iel - ntemp - 1];
      
#pragma omp parallel for private (ii) firstprivate (ahead)
      for (ii = ntemp - 1; ii >= 0; ii += -1) {
        frontier[iel - ii - 1] = frontier[iel - ii - 1] + ahead;
      }
    }
    iel = (nelt / n1 + 1) * n1;
    ntemp1 = iel - nelt;
    if (ntemp1 < ntemp) {
      ahead = frontier[iel - ntemp - 1];
      
#pragma omp parallel for private (ii) firstprivate (ahead,ntemp1,iel)
      for (ii = ntemp - 1; ii >= ntemp1; ii += -1) {
        frontier[iel - ii - 1] = frontier[iel - ii - 1] + ahead;
      }
    }
    ntemp = n1;
  }
}
//------------------------------------------------------------------
// Perform stiffness summation: element-mortar-element mapping
//------------------------------------------------------------------

void dssum()
{
  transfb(dpcmor,((double *)dpcelm));
  transf(dpcmor,((double *)dpcelm));
}
//------------------------------------------------------------------
// assign the value val to face(iface,iel) of array a.
//------------------------------------------------------------------

void facev(double a[5][5][5],int iface,double val)
{
  int kx1;
  int kx2;
  int ky1;
  int ky2;
  int kz1;
  int kz2;
  int ix;
  int iy;
  int iz;
  kx1 = 1;
  ky1 = 1;
  kz1 = 1;
  kx2 = 5;
  ky2 = 5;
  kz2 = 5;
  if (iface == 0) 
    kx1 = 5;
  if (iface == 1) 
    kx2 = 1;
  if (iface == 2) 
    ky1 = 5;
  if (iface == 3) 
    ky2 = 1;
  if (iface == 4) 
    kz1 = 5;
  if (iface == 5) 
    kz2 = 1;
  
#pragma omp parallel for private (ix,iy,iz) firstprivate (kx2,ky1,ky2)
  for (ix = kx1 - 1; ix <= kx2 - 1; ix += 1) {
    
#pragma omp parallel for private (iy,iz) firstprivate (kz1,kz2)
    for (iy = ky1 - 1; iy <= ky2 - 1; iy += 1) {
      
#pragma omp parallel for private (iz) firstprivate (val)
      for (iz = kz1 - 1; iz <= kz2 - 1; iz += 1) {
        a[iz][iy][ix] = val;
      }
    }
  }
}
