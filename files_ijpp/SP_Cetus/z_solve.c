/*
Copyright (C) 1991-2012 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it andor
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http:www.gnu.org/licenses/>. 
*/
/*
This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it. 
*/
/* We do support the IEC 559 math functionality, real and complex.  */
/*
wchar_t uses ISOIEC 10646 (2nd ed., published 2011-03-15) /
   Unicode 6.0. 
*/
/* We do not support C11 <threads.h>.  */
/* ------------------------------------------------------------------------- */
/*                                                                          */
/*  This benchmark is an OpenMP C version of the NPB SP code. This OpenMP   */
/*  C version is developed by the Center for Manycore Programming at Seoul  */
/*  National University and derived from the OpenMP Fortran versions in     */
/*  "NPB3.3-OMP" developed by NAS.                                          */
/*                                                                          */
/*  Permission to use, copy, distribute and modify this software for any    */
/*  purpose with or without fee is hereby granted. This software is         */
/*  provided "as is" without express or implied warranty.                   */
/*                                                                          */
/*  Information on NPB 3.3, including the technical report, the original    */
/*  specifications, source code, results and information on how to submit   */
/*  new results, is available at:                                           */
/*                                                                          */
/*           http:www.nas.nasa.govSoftware/NPB/                          */
/*                                                                          */
/*  Send comments or suggestions for this OpenMP C version to               */
/*  cmp@aces.snu.ac.kr                                                      */
/*                                                                          */
/*          Center for Manycore Programming                                 */
/*          School of Computer Science and Engineering                      */
/*          Seoul National University                                       */
/*          Seoul 151-744, Korea                                            */
/*                                                                          */
/*          E-mail:  cmp@aces.snu.ac.kr                                     */
/*                                                                          */
/* ------------------------------------------------------------------------- */
/* ------------------------------------------------------------------------- */
/* Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,     */
/*          and Jaejin Lee                                                  */
/* ------------------------------------------------------------------------- */
#include "header.h"
/* --------------------------------------------------------------------- */
/* this function performs the solution of the approximate factorization */
/* step in the z-direction for all five matrix components */
/* simultaneously. The Thomas algorithm is employed to solve the */
/* systems for the z-lines. Boundary conditions are non-periodic */
/* --------------------------------------------------------------------- */
void z_solve()
{
	int i, j, k, k1, k2, m;
	double ru1, fac1, fac2;
	/* --------------------------------------------------------------------- */
	/* Prepare for z-solve, array redistribution    */
	/* --------------------------------------------------------------------- */
	int i_tzetar, j_tzetar, k_tzetar;
	double t1_tzetar, t2_tzetar, t3_tzetar, ac_tzetar, xvel_tzetar, yvel_tzetar, zvel_tzetar, r1_tzetar, r2_tzetar, r3_tzetar, r4_tzetar, r5_tzetar;
	double btuz_tzetar, ac2u_tzetar, uzik1_tzetar;
	if (timeron)
	{
		timer_start(8);
	}
	#pragma cetus firstprivate(cv, rhos) 
	#pragma cetus private(fac1, fac2, i, j, k, k1, k2, m, ru1) 
	#pragma cetus lastprivate(cv, rhos) 
	#pragma loop name z_solve#0 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<((((((1L+(-28L*ny2))+((-53L*nx2)*ny2))+((15L*ny2)*grid_points[2L]))+((6L*ny2)*nz2))+(((77L*nx2)*ny2)*grid_points[2L]))+(((31L*nx2)*ny2)*nz2)))) private(fac1, fac2, i, j, k, k1, k2, m, ru1) firstprivate(cv, rhos) lastprivate(cv, rhos)
	for (j=1; j<=ny2; j ++ )
	{
		double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][5];
		double lhsp[(((162/2)*2)+1)][(((162/2)*2)+1)][5];
		double lhsm[(((162/2)*2)+1)][(((162/2)*2)+1)][5];
		/* lhsinitj(nz2+1, nx2); */
		/* lhsinitj - START */
		int i_lhsinitj, m_lhsinitj;
		/* --------------------------------------------------------------------- */
		/* zap the whole left hand side for starters */
		/* set all diagonal values to 1. This is overkill, but convenient */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i_lhsinitj, m_lhsinitj) 
		#pragma loop name z_solve#0#0 
		for (i_lhsinitj=1; i_lhsinitj<=nx2; i_lhsinitj ++ )
		{
			#pragma cetus private(m_lhsinitj) 
			#pragma loop name z_solve#0#0#0 
			for (m_lhsinitj=0; m_lhsinitj<5; m_lhsinitj ++ )
			{
				lhs[0][i_lhsinitj][m_lhsinitj]=0.0;
				lhsp[0][i_lhsinitj][m_lhsinitj]=0.0;
				lhsm[0][i_lhsinitj][m_lhsinitj]=0.0;
				lhs[nz2+1][i_lhsinitj][m_lhsinitj]=0.0;
				lhsp[nz2+1][i_lhsinitj][m_lhsinitj]=0.0;
				lhsm[nz2+1][i_lhsinitj][m_lhsinitj]=0.0;
			}
			lhs[0][i_lhsinitj][2]=1.0;
			lhsp[0][i_lhsinitj][2]=1.0;
			lhsm[0][i_lhsinitj][2]=1.0;
			lhs[nz2+1][i_lhsinitj][2]=1.0;
			lhsp[nz2+1][i_lhsinitj][2]=1.0;
			lhsm[nz2+1][i_lhsinitj][2]=1.0;
		}
		/* lhsinitj - END */
		/* --------------------------------------------------------------------- */
		/* Computes the left hand side for the three z-factors    */
		/* --------------------------------------------------------------------- */
		/* --------------------------------------------------------------------- */
		/* first fill the lhs for the u-eigenvalue                           */
		/* --------------------------------------------------------------------- */
		#pragma cetus firstprivate(cv, rhos) 
		#pragma cetus private(i, k, ru1) 
		#pragma cetus lastprivate(cv, rhos) 
		#pragma loop name z_solve#0#1 
		for (i=1; i<=nx2; i ++ )
		{
			#pragma cetus private(k, ru1) 
			#pragma loop name z_solve#0#1#0 
			for (k=0; k<=(nz2+1); k ++ )
			{
				ru1=(c3c4*rho_i[k][j][i]);
				cv[k]=ws[k][j][i];
				rhos[k]=(((((dz4+(con43*ru1))>(dz5+(c1c5*ru1))) ? (dz4+(con43*ru1)) : (dz5+(c1c5*ru1)))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (((dz4+(con43*ru1))>(dz5+(c1c5*ru1))) ? (dz4+(con43*ru1)) : (dz5+(c1c5*ru1))) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1));
			}
			#pragma cetus private(k) 
			#pragma loop name z_solve#0#1#1 
			for (k=1; k<=nz2; k ++ )
			{
				lhs[k][i][0]=0.0;
				lhs[k][i][1]=((( - dttz2)*cv[k-1])-(dttz1*rhos[k-1]));
				lhs[k][i][2]=(1.0+(c2dttz1*rhos[k]));
				lhs[k][i][3]=((dttz2*cv[k+1])-(dttz1*rhos[k+1]));
				lhs[k][i][4]=0.0;
			}
		}
		/* --------------------------------------------------------------------- */
		/* add fourth order dissipation                                   */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i, k) 
		#pragma loop name z_solve#0#2 
		for (i=1; i<=nx2; i ++ )
		{
			k=1;
			lhs[k][i][2]=(lhs[k][i][2]+comz5);
			lhs[k][i][3]=(lhs[k][i][3]-comz4);
			lhs[k][i][4]=(lhs[k][i][4]+comz1);
			k=2;
			lhs[k][i][1]=(lhs[k][i][1]-comz4);
			lhs[k][i][2]=(lhs[k][i][2]+comz6);
			lhs[k][i][3]=(lhs[k][i][3]-comz4);
			lhs[k][i][4]=(lhs[k][i][4]+comz1);
		}
		#pragma cetus private(i, k) 
		#pragma loop name z_solve#0#3 
		for (k=3; k<=(nz2-2); k ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name z_solve#0#3#0 
			for (i=1; i<=nx2; i ++ )
			{
				lhs[k][i][0]=(lhs[k][i][0]+comz1);
				lhs[k][i][1]=(lhs[k][i][1]-comz4);
				lhs[k][i][2]=(lhs[k][i][2]+comz6);
				lhs[k][i][3]=(lhs[k][i][3]-comz4);
				lhs[k][i][4]=(lhs[k][i][4]+comz1);
			}
		}
		#pragma cetus private(i, k) 
		#pragma loop name z_solve#0#4 
		for (i=1; i<=nx2; i ++ )
		{
			k=(nz2-1);
			lhs[k][i][0]=(lhs[k][i][0]+comz1);
			lhs[k][i][1]=(lhs[k][i][1]-comz4);
			lhs[k][i][2]=(lhs[k][i][2]+comz6);
			lhs[k][i][3]=(lhs[k][i][3]-comz4);
			k=nz2;
			lhs[k][i][0]=(lhs[k][i][0]+comz1);
			lhs[k][i][1]=(lhs[k][i][1]-comz4);
			lhs[k][i][2]=(lhs[k][i][2]+comz5);
		}
		/* --------------------------------------------------------------------- */
		/* subsequently, fill the other factors (u+c), (u-c)  */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i, k) 
		#pragma loop name z_solve#0#5 
		for (k=1; k<=nz2; k ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name z_solve#0#5#0 
			for (i=1; i<=nx2; i ++ )
			{
				lhsp[k][i][0]=lhs[k][i][0];
				lhsp[k][i][1]=(lhs[k][i][1]-(dttz2*speed[k-1][j][i]));
				lhsp[k][i][2]=lhs[k][i][2];
				lhsp[k][i][3]=(lhs[k][i][3]+(dttz2*speed[k+1][j][i]));
				lhsp[k][i][4]=lhs[k][i][4];
				lhsm[k][i][0]=lhs[k][i][0];
				lhsm[k][i][1]=(lhs[k][i][1]+(dttz2*speed[k-1][j][i]));
				lhsm[k][i][2]=lhs[k][i][2];
				lhsm[k][i][3]=(lhs[k][i][3]-(dttz2*speed[k+1][j][i]));
				lhsm[k][i][4]=lhs[k][i][4];
			}
		}
		/* --------------------------------------------------------------------- */
		/* FORWARD ELIMINATION   */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(fac1, i, k, k1, k2, m) 
		#pragma loop name z_solve#0#6 
		for (k=0; k<=(grid_points[2]-3); k ++ )
		{
			k1=(k+1);
			k2=(k+2);
			#pragma cetus private(fac1, i, m) 
			#pragma loop name z_solve#0#6#0 
			for (i=1; i<=nx2; i ++ )
			{
				fac1=(1.0/lhs[k][i][2]);
				lhs[k][i][3]=(fac1*lhs[k][i][3]);
				lhs[k][i][4]=(fac1*lhs[k][i][4]);
				#pragma cetus private(m) 
				#pragma loop name z_solve#0#6#0#0 
				for (m=0; m<3; m ++ )
				{
					rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
				}
				lhs[k1][i][2]=(lhs[k1][i][2]-(lhs[k1][i][1]*lhs[k][i][3]));
				lhs[k1][i][3]=(lhs[k1][i][3]-(lhs[k1][i][1]*lhs[k][i][4]));
				#pragma cetus private(m) 
				#pragma loop name z_solve#0#6#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[k1][j][i][m]=(rhs[k1][j][i][m]-(lhs[k1][i][1]*rhs[k][j][i][m]));
				}
				lhs[k2][i][1]=(lhs[k2][i][1]-(lhs[k2][i][0]*lhs[k][i][3]));
				lhs[k2][i][2]=(lhs[k2][i][2]-(lhs[k2][i][0]*lhs[k][i][4]));
				#pragma cetus private(m) 
				#pragma loop name z_solve#0#6#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[k2][j][i][m]=(rhs[k2][j][i][m]-(lhs[k2][i][0]*rhs[k][j][i][m]));
				}
			}
		}
		/* --------------------------------------------------------------------- */
		/* The last two rows in this grid block are a bit different,  */
		/* since they for (not have two more rows available for the */
		/* elimination of off-diagonal entries */
		/* --------------------------------------------------------------------- */
		k=(grid_points[2]-2);
		k1=(grid_points[2]-1);
		#pragma cetus private(fac1, fac2, i, m) 
		#pragma loop name z_solve#0#7 
		for (i=1; i<=nx2; i ++ )
		{
			fac1=(1.0/lhs[k][i][2]);
			lhs[k][i][3]=(fac1*lhs[k][i][3]);
			lhs[k][i][4]=(fac1*lhs[k][i][4]);
			#pragma cetus private(m) 
			#pragma loop name z_solve#0#7#0 
			for (m=0; m<3; m ++ )
			{
				rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
			}
			lhs[k1][i][2]=(lhs[k1][i][2]-(lhs[k1][i][1]*lhs[k][i][3]));
			lhs[k1][i][3]=(lhs[k1][i][3]-(lhs[k1][i][1]*lhs[k][i][4]));
			#pragma cetus private(m) 
			#pragma loop name z_solve#0#7#1 
			for (m=0; m<3; m ++ )
			{
				rhs[k1][j][i][m]=(rhs[k1][j][i][m]-(lhs[k1][i][1]*rhs[k][j][i][m]));
			}
			/* --------------------------------------------------------------------- */
			/* scale the last row immediately */
			/* --------------------------------------------------------------------- */
			fac2=(1.0/lhs[k1][i][2]);
			#pragma cetus private(m) 
			#pragma loop name z_solve#0#7#2 
			for (m=0; m<3; m ++ )
			{
				rhs[k1][j][i][m]=(fac2*rhs[k1][j][i][m]);
			}
		}
		/* --------------------------------------------------------------------- */
		/* for (the u+c and the u-c factors                */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(fac1, i, k, k1, k2, m) 
		#pragma loop name z_solve#0#8 
		for (k=0; k<=(grid_points[2]-3); k ++ )
		{
			k1=(k+1);
			k2=(k+2);
			#pragma cetus private(fac1, i, m) 
			#pragma loop name z_solve#0#8#0 
			for (i=1; i<=nx2; i ++ )
			{
				m=3;
				fac1=(1.0/lhsp[k][i][2]);
				lhsp[k][i][3]=(fac1*lhsp[k][i][3]);
				lhsp[k][i][4]=(fac1*lhsp[k][i][4]);
				rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
				lhsp[k1][i][2]=(lhsp[k1][i][2]-(lhsp[k1][i][1]*lhsp[k][i][3]));
				lhsp[k1][i][3]=(lhsp[k1][i][3]-(lhsp[k1][i][1]*lhsp[k][i][4]));
				rhs[k1][j][i][m]=(rhs[k1][j][i][m]-(lhsp[k1][i][1]*rhs[k][j][i][m]));
				lhsp[k2][i][1]=(lhsp[k2][i][1]-(lhsp[k2][i][0]*lhsp[k][i][3]));
				lhsp[k2][i][2]=(lhsp[k2][i][2]-(lhsp[k2][i][0]*lhsp[k][i][4]));
				rhs[k2][j][i][m]=(rhs[k2][j][i][m]-(lhsp[k2][i][0]*rhs[k][j][i][m]));
				m=4;
				fac1=(1.0/lhsm[k][i][2]);
				lhsm[k][i][3]=(fac1*lhsm[k][i][3]);
				lhsm[k][i][4]=(fac1*lhsm[k][i][4]);
				rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
				lhsm[k1][i][2]=(lhsm[k1][i][2]-(lhsm[k1][i][1]*lhsm[k][i][3]));
				lhsm[k1][i][3]=(lhsm[k1][i][3]-(lhsm[k1][i][1]*lhsm[k][i][4]));
				rhs[k1][j][i][m]=(rhs[k1][j][i][m]-(lhsm[k1][i][1]*rhs[k][j][i][m]));
				lhsm[k2][i][1]=(lhsm[k2][i][1]-(lhsm[k2][i][0]*lhsm[k][i][3]));
				lhsm[k2][i][2]=(lhsm[k2][i][2]-(lhsm[k2][i][0]*lhsm[k][i][4]));
				rhs[k2][j][i][m]=(rhs[k2][j][i][m]-(lhsm[k2][i][0]*rhs[k][j][i][m]));
			}
		}
		/* --------------------------------------------------------------------- */
		/* And again the last two rows separately */
		/* --------------------------------------------------------------------- */
		k=(grid_points[2]-2);
		k1=(grid_points[2]-1);
		#pragma cetus private(fac1, i, m) 
		#pragma loop name z_solve#0#9 
		for (i=1; i<=nx2; i ++ )
		{
			m=3;
			fac1=(1.0/lhsp[k][i][2]);
			lhsp[k][i][3]=(fac1*lhsp[k][i][3]);
			lhsp[k][i][4]=(fac1*lhsp[k][i][4]);
			rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
			lhsp[k1][i][2]=(lhsp[k1][i][2]-(lhsp[k1][i][1]*lhsp[k][i][3]));
			lhsp[k1][i][3]=(lhsp[k1][i][3]-(lhsp[k1][i][1]*lhsp[k][i][4]));
			rhs[k1][j][i][m]=(rhs[k1][j][i][m]-(lhsp[k1][i][1]*rhs[k][j][i][m]));
			m=4;
			fac1=(1.0/lhsm[k][i][2]);
			lhsm[k][i][3]=(fac1*lhsm[k][i][3]);
			lhsm[k][i][4]=(fac1*lhsm[k][i][4]);
			rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
			lhsm[k1][i][2]=(lhsm[k1][i][2]-(lhsm[k1][i][1]*lhsm[k][i][3]));
			lhsm[k1][i][3]=(lhsm[k1][i][3]-(lhsm[k1][i][1]*lhsm[k][i][4]));
			rhs[k1][j][i][m]=(rhs[k1][j][i][m]-(lhsm[k1][i][1]*rhs[k][j][i][m]));
			/* --------------------------------------------------------------------- */
			/* Scale the last row immediately (some of this is overkill */
			/* if this is the last cell) */
			/* --------------------------------------------------------------------- */
			rhs[k1][j][i][3]=(rhs[k1][j][i][3]/lhsp[k1][i][2]);
			rhs[k1][j][i][4]=(rhs[k1][j][i][4]/lhsm[k1][i][2]);
		}
		/* --------------------------------------------------------------------- */
		/* BACKSUBSTITUTION  */
		/* --------------------------------------------------------------------- */
		k=(grid_points[2]-2);
		k1=(grid_points[2]-1);
		#pragma cetus private(i, m) 
		#pragma loop name z_solve#0#10 
		for (i=1; i<=nx2; i ++ )
		{
			#pragma cetus private(m) 
			#pragma loop name z_solve#0#10#0 
			for (m=0; m<3; m ++ )
			{
				rhs[k][j][i][m]=(rhs[k][j][i][m]-(lhs[k][i][3]*rhs[k1][j][i][m]));
			}
			rhs[k][j][i][3]=(rhs[k][j][i][3]-(lhsp[k][i][3]*rhs[k1][j][i][3]));
			rhs[k][j][i][4]=(rhs[k][j][i][4]-(lhsm[k][i][3]*rhs[k1][j][i][4]));
		}
		/* --------------------------------------------------------------------- */
		/* Whether or not this is the last processor, we always have */
		/* to complete the back-substitution  */
		/* --------------------------------------------------------------------- */
		/* --------------------------------------------------------------------- */
		/* The first three factors */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i, k, k1, k2, m) 
		#pragma loop name z_solve#0#11 
		for (k=(grid_points[2]-3); k>=0; k -- )
		{
			k1=(k+1);
			k2=(k+2);
			#pragma cetus private(i, m) 
			#pragma loop name z_solve#0#11#0 
			for (i=1; i<=nx2; i ++ )
			{
				#pragma cetus private(m) 
				#pragma loop name z_solve#0#11#0#0 
				for (m=0; m<3; m ++ )
				{
					rhs[k][j][i][m]=((rhs[k][j][i][m]-(lhs[k][i][3]*rhs[k1][j][i][m]))-(lhs[k][i][4]*rhs[k2][j][i][m]));
				}
				/* ------------------------------------------------------------------- */
				/* And the remaining two */
				/* ------------------------------------------------------------------- */
				rhs[k][j][i][3]=((rhs[k][j][i][3]-(lhsp[k][i][3]*rhs[k1][j][i][3]))-(lhsp[k][i][4]*rhs[k2][j][i][3]));
				rhs[k][j][i][4]=((rhs[k][j][i][4]-(lhsm[k][i][3]*rhs[k1][j][i][4]))-(lhsm[k][i][4]*rhs[k2][j][i][4]));
			}
		}
	}
	if (timeron)
	{
		timer_stop(8);
	}
	/* tzetar(); */
	/* tzetar - START */
	if (timeron)
	{
		timer_start(14);
	}
	#pragma cetus private(ac2u_tzetar, ac_tzetar, btuz_tzetar, i_tzetar, j_tzetar, k_tzetar, r1_tzetar, r2_tzetar, r3_tzetar, r4_tzetar, r5_tzetar, t1_tzetar, t2_tzetar, t3_tzetar, uzik1_tzetar, xvel_tzetar, yvel_tzetar, zvel_tzetar) 
	#pragma loop name z_solve#1 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(((1L+(3L*nz2))+((3L*ny2)*nz2))+(((22L*nx2)*ny2)*nz2)))) private(ac2u_tzetar, ac_tzetar, btuz_tzetar, i_tzetar, j_tzetar, k_tzetar, r1_tzetar, r2_tzetar, r3_tzetar, r4_tzetar, r5_tzetar, t1_tzetar, t2_tzetar, t3_tzetar, uzik1_tzetar, xvel_tzetar, yvel_tzetar, zvel_tzetar)
	for (k_tzetar=1; k_tzetar<=nz2; k_tzetar ++ )
	{
		#pragma cetus private(ac2u_tzetar, ac_tzetar, btuz_tzetar, i_tzetar, j_tzetar, r1_tzetar, r2_tzetar, r3_tzetar, r4_tzetar, r5_tzetar, t1_tzetar, t2_tzetar, t3_tzetar, uzik1_tzetar, xvel_tzetar, yvel_tzetar, zvel_tzetar) 
		#pragma loop name z_solve#1#0 
		for (j_tzetar=1; j_tzetar<=ny2; j_tzetar ++ )
		{
			#pragma cetus private(ac2u_tzetar, ac_tzetar, btuz_tzetar, i_tzetar, r1_tzetar, r2_tzetar, r3_tzetar, r4_tzetar, r5_tzetar, t1_tzetar, t2_tzetar, t3_tzetar, uzik1_tzetar, xvel_tzetar, yvel_tzetar, zvel_tzetar) 
			#pragma loop name z_solve#1#0#0 
			for (i_tzetar=1; i_tzetar<=nx2; i_tzetar ++ )
			{
				xvel_tzetar=us[k_tzetar][j_tzetar][i_tzetar];
				yvel_tzetar=vs[k_tzetar][j_tzetar][i_tzetar];
				zvel_tzetar=ws[k_tzetar][j_tzetar][i_tzetar];
				ac_tzetar=speed[k_tzetar][j_tzetar][i_tzetar];
				ac2u_tzetar=(ac_tzetar*ac_tzetar);
				r1_tzetar=rhs[k_tzetar][j_tzetar][i_tzetar][0];
				r2_tzetar=rhs[k_tzetar][j_tzetar][i_tzetar][1];
				r3_tzetar=rhs[k_tzetar][j_tzetar][i_tzetar][2];
				r4_tzetar=rhs[k_tzetar][j_tzetar][i_tzetar][3];
				r5_tzetar=rhs[k_tzetar][j_tzetar][i_tzetar][4];
				uzik1_tzetar=u[k_tzetar][j_tzetar][i_tzetar][0];
				btuz_tzetar=(bt*uzik1_tzetar);
				t1_tzetar=((btuz_tzetar/ac_tzetar)*(r4_tzetar+r5_tzetar));
				t2_tzetar=(r3_tzetar+t1_tzetar);
				t3_tzetar=(btuz_tzetar*(r4_tzetar-r5_tzetar));
				rhs[k_tzetar][j_tzetar][i_tzetar][0]=t2_tzetar;
				rhs[k_tzetar][j_tzetar][i_tzetar][1]=((( - uzik1_tzetar)*r2_tzetar)+(xvel_tzetar*t2_tzetar));
				rhs[k_tzetar][j_tzetar][i_tzetar][2]=((uzik1_tzetar*r1_tzetar)+(yvel_tzetar*t2_tzetar));
				rhs[k_tzetar][j_tzetar][i_tzetar][3]=((zvel_tzetar*t2_tzetar)+t3_tzetar);
				rhs[k_tzetar][j_tzetar][i_tzetar][4]=((((uzik1_tzetar*((( - xvel_tzetar)*r2_tzetar)+(yvel_tzetar*r1_tzetar)))+(qs[k_tzetar][j_tzetar][i_tzetar]*t2_tzetar))+((c2iv*ac2u_tzetar)*t1_tzetar))+(zvel_tzetar*t3_tzetar));
			}
		}
	}
	if (timeron)
	{
		timer_stop(14);
	}
	/* tzetar - END */
}
