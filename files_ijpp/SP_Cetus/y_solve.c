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
/* step in the y-direction for all five matrix components */
/* simultaneously. The Thomas algorithm is employed to solve the */
/* systems for the y-lines. Boundary conditions are non-periodic */
/* --------------------------------------------------------------------- */
void y_solve()
{
	int i, j, k, j1, j2, m;
	double ru1, fac1, fac2;
	int i_pinvr, j_pinvr, k_pinvr;
	double r1_pinvr, r2_pinvr, r3_pinvr, r4_pinvr, r5_pinvr, t1_pinvr, t2_pinvr;
	if (timeron)
	{
		timer_start(7);
	}
	#pragma cetus firstprivate(cv, rhoq) 
	#pragma cetus private(fac1, fac2, i, j, j1, j2, k, ru1) 
	#pragma cetus lastprivate(cv, rhoq) 
	#pragma loop name y_solve#0 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(((((1L+(292L*nz2))+((49L*nx2)*nz2))+((-166L*nz2)*grid_points[0L]))+((-195L*nz2)*grid_points[1L]))+(((108L*nz2)*grid_points[0L])*grid_points[1L])))) private(fac1, fac2, i, j, j1, j2, k, ru1) firstprivate(cv, rhoq) lastprivate(cv, rhoq)
	for (k=1; k<=nz2; k ++ )
	{
		double lhs[(((162/2)*2)+1)][(((162/2)*2)+1)][5];
		double lhsp[(((162/2)*2)+1)][(((162/2)*2)+1)][5];
		double lhsm[(((162/2)*2)+1)][(((162/2)*2)+1)][5];
		/* lhsinitj(ny2+1, nx2); */
		/* lhsinitj - START */
		int i_lhsinitj, m;
		/* --------------------------------------------------------------------- */
		/* zap the whole left hand side for starters */
		/* set all diagonal values to 1. This is overkill, but convenient */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i_lhsinitj, m) 
		#pragma loop name y_solve#0#0 
		for (i_lhsinitj=1; i_lhsinitj<=nx2; i_lhsinitj ++ )
		{
			#pragma cetus private(m) 
			#pragma loop name y_solve#0#0#0 
			for (m=0; m<5; m ++ )
			{
				lhs[0][i_lhsinitj][m]=0.0;
				lhsp[0][i_lhsinitj][m]=0.0;
				lhsm[0][i_lhsinitj][m]=0.0;
				lhs[ny2+1][i_lhsinitj][m]=0.0;
				lhsp[ny2+1][i_lhsinitj][m]=0.0;
				lhsm[ny2+1][i_lhsinitj][m]=0.0;
			}
			lhs[0][i_lhsinitj][2]=1.0;
			lhsp[0][i_lhsinitj][2]=1.0;
			lhsm[0][i_lhsinitj][2]=1.0;
			lhs[ny2+1][i_lhsinitj][2]=1.0;
			lhsp[ny2+1][i_lhsinitj][2]=1.0;
			lhsm[ny2+1][i_lhsinitj][2]=1.0;
		}
		/* lhsinitj - END */
		/* --------------------------------------------------------------------- */
		/* Computes the left hand side for the three y-factors    */
		/* --------------------------------------------------------------------- */
		/* --------------------------------------------------------------------- */
		/* first fill the lhs for the u-eigenvalue          */
		/* --------------------------------------------------------------------- */
		#pragma cetus firstprivate(cv, rhoq) 
		#pragma cetus private(i, j, ru1) 
		#pragma cetus lastprivate(cv, rhoq) 
		#pragma loop name y_solve#0#1 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma cetus private(j, ru1) 
			#pragma loop name y_solve#0#1#0 
			for (j=0; j<=(grid_points[1]-1); j ++ )
			{
				ru1=(c3c4*rho_i[k][j][i]);
				cv[j]=vs[k][j][i];
				rhoq[j]=(((((dy3+(con43*ru1))>(dy5+(c1c5*ru1))) ? (dy3+(con43*ru1)) : (dy5+(c1c5*ru1)))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (((dy3+(con43*ru1))>(dy5+(c1c5*ru1))) ? (dy3+(con43*ru1)) : (dy5+(c1c5*ru1))) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1));
			}
			#pragma cetus private(j) 
			#pragma loop name y_solve#0#1#1 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				lhs[j][i][0]=0.0;
				lhs[j][i][1]=((( - dtty2)*cv[j-1])-(dtty1*rhoq[j-1]));
				lhs[j][i][2]=(1.0+(c2dtty1*rhoq[j]));
				lhs[j][i][3]=((dtty2*cv[j+1])-(dtty1*rhoq[j+1]));
				lhs[j][i][4]=0.0;
			}
		}
		/* --------------------------------------------------------------------- */
		/* add fourth order dissipation                              */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i, j) 
		#pragma loop name y_solve#0#2 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			j=1;
			lhs[j][i][2]=(lhs[j][i][2]+comz5);
			lhs[j][i][3]=(lhs[j][i][3]-comz4);
			lhs[j][i][4]=(lhs[j][i][4]+comz1);
			lhs[j+1][i][1]=(lhs[j+1][i][1]-comz4);
			lhs[j+1][i][2]=(lhs[j+1][i][2]+comz6);
			lhs[j+1][i][3]=(lhs[j+1][i][3]-comz4);
			lhs[j+1][i][4]=(lhs[j+1][i][4]+comz1);
		}
		#pragma cetus private(i, j) 
		#pragma loop name y_solve#0#3 
		for (j=3; j<=(grid_points[1]-4); j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name y_solve#0#3#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				lhs[j][i][0]=(lhs[j][i][0]+comz1);
				lhs[j][i][1]=(lhs[j][i][1]-comz4);
				lhs[j][i][2]=(lhs[j][i][2]+comz6);
				lhs[j][i][3]=(lhs[j][i][3]-comz4);
				lhs[j][i][4]=(lhs[j][i][4]+comz1);
			}
		}
		#pragma cetus private(i, j) 
		#pragma loop name y_solve#0#4 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			j=(grid_points[1]-3);
			lhs[j][i][0]=(lhs[j][i][0]+comz1);
			lhs[j][i][1]=(lhs[j][i][1]-comz4);
			lhs[j][i][2]=(lhs[j][i][2]+comz6);
			lhs[j][i][3]=(lhs[j][i][3]-comz4);
			lhs[j+1][i][0]=(lhs[j+1][i][0]+comz1);
			lhs[j+1][i][1]=(lhs[j+1][i][1]-comz4);
			lhs[j+1][i][2]=(lhs[j+1][i][2]+comz5);
		}
		/* --------------------------------------------------------------------- */
		/* subsequently, for (the other two factors                     */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i, j) 
		#pragma loop name y_solve#0#5 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name y_solve#0#5#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				lhsp[j][i][0]=lhs[j][i][0];
				lhsp[j][i][1]=(lhs[j][i][1]-(dtty2*speed[k][j-1][i]));
				lhsp[j][i][2]=lhs[j][i][2];
				lhsp[j][i][3]=(lhs[j][i][3]+(dtty2*speed[k][j+1][i]));
				lhsp[j][i][4]=lhs[j][i][4];
				lhsm[j][i][0]=lhs[j][i][0];
				lhsm[j][i][1]=(lhs[j][i][1]+(dtty2*speed[k][j-1][i]));
				lhsm[j][i][2]=lhs[j][i][2];
				lhsm[j][i][3]=(lhs[j][i][3]-(dtty2*speed[k][j+1][i]));
				lhsm[j][i][4]=lhs[j][i][4];
			}
		}
		/* --------------------------------------------------------------------- */
		/* FORWARD ELIMINATION   */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(fac1, i, j, j1, j2, m) 
		#pragma loop name y_solve#0#6 
		for (j=0; j<=(grid_points[1]-3); j ++ )
		{
			j1=(j+1);
			j2=(j+2);
			#pragma cetus private(fac1, i, m) 
			#pragma loop name y_solve#0#6#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				fac1=(1.0/lhs[j][i][2]);
				lhs[j][i][3]=(fac1*lhs[j][i][3]);
				lhs[j][i][4]=(fac1*lhs[j][i][4]);
				#pragma cetus private(m) 
				#pragma loop name y_solve#0#6#0#0 
				for (m=0; m<3; m ++ )
				{
					rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
				}
				lhs[j1][i][2]=(lhs[j1][i][2]-(lhs[j1][i][1]*lhs[j][i][3]));
				lhs[j1][i][3]=(lhs[j1][i][3]-(lhs[j1][i][1]*lhs[j][i][4]));
				#pragma cetus private(m) 
				#pragma loop name y_solve#0#6#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[k][j1][i][m]=(rhs[k][j1][i][m]-(lhs[j1][i][1]*rhs[k][j][i][m]));
				}
				lhs[j2][i][1]=(lhs[j2][i][1]-(lhs[j2][i][0]*lhs[j][i][3]));
				lhs[j2][i][2]=(lhs[j2][i][2]-(lhs[j2][i][0]*lhs[j][i][4]));
				#pragma cetus private(m) 
				#pragma loop name y_solve#0#6#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[k][j2][i][m]=(rhs[k][j2][i][m]-(lhs[j2][i][0]*rhs[k][j][i][m]));
				}
			}
		}
		/* --------------------------------------------------------------------- */
		/* The last two rows in this grid block are a bit different,  */
		/* since they for (not have two more rows available for the */
		/* elimination of off-diagonal entries */
		/* --------------------------------------------------------------------- */
		j=(grid_points[1]-2);
		j1=(grid_points[1]-1);
		#pragma cetus private(fac1, fac2, i, m) 
		#pragma loop name y_solve#0#7 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			fac1=(1.0/lhs[j][i][2]);
			lhs[j][i][3]=(fac1*lhs[j][i][3]);
			lhs[j][i][4]=(fac1*lhs[j][i][4]);
			#pragma cetus private(m) 
			#pragma loop name y_solve#0#7#0 
			for (m=0; m<3; m ++ )
			{
				rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
			}
			lhs[j1][i][2]=(lhs[j1][i][2]-(lhs[j1][i][1]*lhs[j][i][3]));
			lhs[j1][i][3]=(lhs[j1][i][3]-(lhs[j1][i][1]*lhs[j][i][4]));
			#pragma cetus private(m) 
			#pragma loop name y_solve#0#7#1 
			for (m=0; m<3; m ++ )
			{
				rhs[k][j1][i][m]=(rhs[k][j1][i][m]-(lhs[j1][i][1]*rhs[k][j][i][m]));
			}
			/* --------------------------------------------------------------------- */
			/* scale the last row immediately  */
			/* --------------------------------------------------------------------- */
			fac2=(1.0/lhs[j1][i][2]);
			#pragma cetus private(m) 
			#pragma loop name y_solve#0#7#2 
			for (m=0; m<3; m ++ )
			{
				rhs[k][j1][i][m]=(fac2*rhs[k][j1][i][m]);
			}
		}
		/* --------------------------------------------------------------------- */
		/* for (the u+c and the u-c factors                  */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(fac1, i, j, j1, j2, m) 
		#pragma loop name y_solve#0#8 
		for (j=0; j<=(grid_points[1]-3); j ++ )
		{
			j1=(j+1);
			j2=(j+2);
			#pragma cetus private(fac1, i, m) 
			#pragma loop name y_solve#0#8#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				m=3;
				fac1=(1.0/lhsp[j][i][2]);
				lhsp[j][i][3]=(fac1*lhsp[j][i][3]);
				lhsp[j][i][4]=(fac1*lhsp[j][i][4]);
				rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
				lhsp[j1][i][2]=(lhsp[j1][i][2]-(lhsp[j1][i][1]*lhsp[j][i][3]));
				lhsp[j1][i][3]=(lhsp[j1][i][3]-(lhsp[j1][i][1]*lhsp[j][i][4]));
				rhs[k][j1][i][m]=(rhs[k][j1][i][m]-(lhsp[j1][i][1]*rhs[k][j][i][m]));
				lhsp[j2][i][1]=(lhsp[j2][i][1]-(lhsp[j2][i][0]*lhsp[j][i][3]));
				lhsp[j2][i][2]=(lhsp[j2][i][2]-(lhsp[j2][i][0]*lhsp[j][i][4]));
				rhs[k][j2][i][m]=(rhs[k][j2][i][m]-(lhsp[j2][i][0]*rhs[k][j][i][m]));
				m=4;
				fac1=(1.0/lhsm[j][i][2]);
				lhsm[j][i][3]=(fac1*lhsm[j][i][3]);
				lhsm[j][i][4]=(fac1*lhsm[j][i][4]);
				rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
				lhsm[j1][i][2]=(lhsm[j1][i][2]-(lhsm[j1][i][1]*lhsm[j][i][3]));
				lhsm[j1][i][3]=(lhsm[j1][i][3]-(lhsm[j1][i][1]*lhsm[j][i][4]));
				rhs[k][j1][i][m]=(rhs[k][j1][i][m]-(lhsm[j1][i][1]*rhs[k][j][i][m]));
				lhsm[j2][i][1]=(lhsm[j2][i][1]-(lhsm[j2][i][0]*lhsm[j][i][3]));
				lhsm[j2][i][2]=(lhsm[j2][i][2]-(lhsm[j2][i][0]*lhsm[j][i][4]));
				rhs[k][j2][i][m]=(rhs[k][j2][i][m]-(lhsm[j2][i][0]*rhs[k][j][i][m]));
			}
		}
		/* --------------------------------------------------------------------- */
		/* And again the last two rows separately */
		/* --------------------------------------------------------------------- */
		j=(grid_points[1]-2);
		j1=(grid_points[1]-1);
		#pragma cetus private(fac1, i, m) 
		#pragma loop name y_solve#0#9 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			m=3;
			fac1=(1.0/lhsp[j][i][2]);
			lhsp[j][i][3]=(fac1*lhsp[j][i][3]);
			lhsp[j][i][4]=(fac1*lhsp[j][i][4]);
			rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
			lhsp[j1][i][2]=(lhsp[j1][i][2]-(lhsp[j1][i][1]*lhsp[j][i][3]));
			lhsp[j1][i][3]=(lhsp[j1][i][3]-(lhsp[j1][i][1]*lhsp[j][i][4]));
			rhs[k][j1][i][m]=(rhs[k][j1][i][m]-(lhsp[j1][i][1]*rhs[k][j][i][m]));
			m=4;
			fac1=(1.0/lhsm[j][i][2]);
			lhsm[j][i][3]=(fac1*lhsm[j][i][3]);
			lhsm[j][i][4]=(fac1*lhsm[j][i][4]);
			rhs[k][j][i][m]=(fac1*rhs[k][j][i][m]);
			lhsm[j1][i][2]=(lhsm[j1][i][2]-(lhsm[j1][i][1]*lhsm[j][i][3]));
			lhsm[j1][i][3]=(lhsm[j1][i][3]-(lhsm[j1][i][1]*lhsm[j][i][4]));
			rhs[k][j1][i][m]=(rhs[k][j1][i][m]-(lhsm[j1][i][1]*rhs[k][j][i][m]));
			/* --------------------------------------------------------------------- */
			/* Scale the last row immediately  */
			/* --------------------------------------------------------------------- */
			rhs[k][j1][i][3]=(rhs[k][j1][i][3]/lhsp[j1][i][2]);
			rhs[k][j1][i][4]=(rhs[k][j1][i][4]/lhsm[j1][i][2]);
		}
		/* --------------------------------------------------------------------- */
		/* BACKSUBSTITUTION  */
		/* --------------------------------------------------------------------- */
		j=(grid_points[1]-2);
		j1=(grid_points[1]-1);
		#pragma cetus private(i, m) 
		#pragma loop name y_solve#0#10 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma cetus private(m) 
			#pragma loop name y_solve#0#10#0 
			for (m=0; m<3; m ++ )
			{
				rhs[k][j][i][m]=(rhs[k][j][i][m]-(lhs[j][i][3]*rhs[k][j1][i][m]));
			}
			rhs[k][j][i][3]=(rhs[k][j][i][3]-(lhsp[j][i][3]*rhs[k][j1][i][3]));
			rhs[k][j][i][4]=(rhs[k][j][i][4]-(lhsm[j][i][3]*rhs[k][j1][i][4]));
		}
		/* --------------------------------------------------------------------- */
		/* The first three factors */
		/* --------------------------------------------------------------------- */
		#pragma cetus private(i, j, j1, j2, m) 
		#pragma loop name y_solve#0#11 
		for (j=(grid_points[1]-3); j>=0; j -- )
		{
			j1=(j+1);
			j2=(j+2);
			#pragma cetus private(i, m) 
			#pragma loop name y_solve#0#11#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				#pragma cetus private(m) 
				#pragma loop name y_solve#0#11#0#0 
				for (m=0; m<3; m ++ )
				{
					rhs[k][j][i][m]=((rhs[k][j][i][m]-(lhs[j][i][3]*rhs[k][j1][i][m]))-(lhs[j][i][4]*rhs[k][j2][i][m]));
				}
				/* ------------------------------------------------------------------- */
				/* And the remaining two */
				/* ------------------------------------------------------------------- */
				rhs[k][j][i][3]=((rhs[k][j][i][3]-(lhsp[j][i][3]*rhs[k][j1][i][3]))-(lhsp[j][i][4]*rhs[k][j2][i][3]));
				rhs[k][j][i][4]=((rhs[k][j][i][4]-(lhsm[j][i][3]*rhs[k][j1][i][4]))-(lhsm[j][i][4]*rhs[k][j2][i][4]));
			}
		}
	}
	if (timeron)
	{
		timer_stop(7);
	}
	/* pinvr(); */
	/* pinvr - START */
	if (timeron)
	{
		timer_start(12);
	}
	#pragma cetus private(i_pinvr, j_pinvr, k_pinvr, r1_pinvr, r2_pinvr, r3_pinvr, r4_pinvr, r5_pinvr, t1_pinvr, t2_pinvr) 
	#pragma loop name y_solve#1 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(((1L+(3L*nz2))+((3L*ny2)*nz2))+(((14L*nx2)*ny2)*nz2)))) private(i_pinvr, j_pinvr, k_pinvr, r1_pinvr, r2_pinvr, r3_pinvr, r4_pinvr, r5_pinvr, t1_pinvr, t2_pinvr)
	for (k_pinvr=1; k_pinvr<=nz2; k_pinvr ++ )
	{
		#pragma cetus private(i_pinvr, j_pinvr, r1_pinvr, r2_pinvr, r3_pinvr, r4_pinvr, r5_pinvr, t1_pinvr, t2_pinvr) 
		#pragma loop name y_solve#1#0 
		for (j_pinvr=1; j_pinvr<=ny2; j_pinvr ++ )
		{
			#pragma cetus private(i_pinvr, r1_pinvr, r2_pinvr, r3_pinvr, r4_pinvr, r5_pinvr, t1_pinvr, t2_pinvr) 
			#pragma loop name y_solve#1#0#0 
			for (i_pinvr=1; i_pinvr<=nx2; i_pinvr ++ )
			{
				r1_pinvr=rhs[k_pinvr][j_pinvr][i_pinvr][0];
				r2_pinvr=rhs[k_pinvr][j_pinvr][i_pinvr][1];
				r3_pinvr=rhs[k_pinvr][j_pinvr][i_pinvr][2];
				r4_pinvr=rhs[k_pinvr][j_pinvr][i_pinvr][3];
				r5_pinvr=rhs[k_pinvr][j_pinvr][i_pinvr][4];
				t1_pinvr=(bt*r1_pinvr);
				t2_pinvr=(0.5*(r4_pinvr+r5_pinvr));
				rhs[k_pinvr][j_pinvr][i_pinvr][0]=(bt*(r4_pinvr-r5_pinvr));
				rhs[k_pinvr][j_pinvr][i_pinvr][1]=( - r3_pinvr);
				rhs[k_pinvr][j_pinvr][i_pinvr][2]=r2_pinvr;
				rhs[k_pinvr][j_pinvr][i_pinvr][3]=(( - t1_pinvr)+t2_pinvr);
				rhs[k_pinvr][j_pinvr][i_pinvr][4]=(t1_pinvr+t2_pinvr);
			}
		}
	}
	if (timeron)
	{
		timer_stop(12);
	}
	/* pinvr - END */
}
