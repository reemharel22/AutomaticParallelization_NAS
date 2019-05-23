#include <stdlib.h>
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
/*  This benchmark is an OpenMP C version of the NPB UA code. This OpenMP   */
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
/* ------------------------------------------------------------------ */
/* This version uses ATOMIC for atomic updates,  */
/* but locks are still used in get_emo (mason.c). */
/* ------------------------------------------------------------------ */
void init_locks()
{
	int i;
	/* initialize locks in parallel */
	#pragma cetus private(i) 
	#pragma loop name init_locks#0 
	for (i=0; i<(8*33500); i ++ )
	{
		omp_init_lock( & tlock[i]);
	}
}

/* ------------------------------------------------------------------ */
/* Map values from mortar(tmor) to element(tx) */
/* ------------------------------------------------------------------ */
void transf(double tmor[], double tx[])
{
	double tmp[2][5][5];
	int ig1, ig2, ig3, ig4, ie, iface, il1, il2, il3, il4;
	int nnje, ije1, ije2, col, i, j, ig, il;
	/* zero out tx on element boundaries */
	col2(tx, (double * )tmult, ntot);
	#pragma cetus private(col, i, ie, iface, ig, ig1, ig2, ig3, ig4, ije1, ije2, il, il1, il2, il3, il4, j, nnje) 
	#pragma loop name transf#0 
	for (ie=0; ie<nelt; ie ++ )
	{
		#pragma cetus private(col, i, iface, ig, ig1, ig2, ig3, ig4, ije1, ije2, il, il1, il2, il3, il4, j, nnje) 
		#pragma loop name transf#0#0 
		for (iface=0; iface<6; iface ++ )
		{
			/* get the collocation point index of the four local corners on the */
			/* face iface of element ie */
			il1=idel[ie][iface][0][0];
			il2=idel[ie][iface][0][5-1];
			il3=idel[ie][iface][5-1][0];
			il4=idel[ie][iface][5-1][5-1];
			/* get the mortar indices of the four local corners */
			ig1=idmo[ie][iface][0][0][0][0];
			ig2=idmo[ie][iface][1][0][0][5-1];
			ig3=idmo[ie][iface][0][1][5-1][0];
			ig4=idmo[ie][iface][1][1][5-1][5-1];
			/* copy the value from tmor to tx for these four local corners */
			tx[il1]=tmor[ig1];
			tx[il2]=tmor[ig2];
			tx[il3]=tmor[ig3];
			tx[il4]=tmor[ig4];
			/* nnje=1 for conforming faces, nnje=2 for nonconforming faces */
			if (cbc[ie][iface]==3)
			{
				nnje=2;
			}
			else
			{
				nnje=1;
			}
			/* for nonconforming faces */
			if (nnje==2)
			{
				/* nonconforming faces have four pieces of mortar, first map them to */
				/* two intermediate mortars, stored in tmp */
				r_init((double * )tmp, (5*5)*2, 0.0);
				#pragma cetus private(col, i, ig, ije1, ije2, il, j) 
				#pragma loop name transf#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for if((10000<((1L+(3L*nnje))+((393L*nnje)*nnje)))) private(col, i, ig, ije1, ije2, il, j)
				for (ije1=0; ije1<nnje; ije1 ++ )
				{
					#pragma cetus private(col, i, ig, ije2, il, j) 
					#pragma loop name transf#0#0#0#0 
					for (ije2=0; ije2<nnje; ije2 ++ )
					{
						#pragma cetus private(col, i, ig, il, j) 
						#pragma loop name transf#0#0#0#0#0 
						for (col=0; col<5; col ++ )
						{
							/* in each row col, when coloumn i=1 or LX1, the value */
							/* in tmor is copied to tmp */
							i=v_end[ije2];
							ig=idmo[ie][iface][ije2][ije1][col][i];
							tmp[ije1][col][i]=tmor[ig];
							/* in each row col, value in the interior three collocation */
							/* points is computed by apply mapping matrix qbnew to tmor */
							#pragma cetus private(i, ig, il, j) 
							#pragma loop name transf#0#0#0#0#0#0 
							for (i=1; i<(5-1); i ++ )
							{
								il=idel[ie][iface][col][i];
								#pragma cetus private(ig, j) 
								#pragma loop name transf#0#0#0#0#0#0#0 
								/* #pragma cetus reduction(+: tmp[ije1][col][i])  */
								for (j=0; j<5; j ++ )
								{
									ig=idmo[ie][iface][ije2][ije1][col][j];
									tmp[ije1][col][i]=(tmp[ije1][col][i]+(qbnew[ije2][j][i-1]*tmor[ig]));
								}
							}
						}
					}
				}
				/* mapping from two pieces of intermediate mortar tmp to element */
				/* face tx */
				#pragma cetus private(col, i, ije1, il, j) 
				#pragma loop name transf#0#0#1 
				/* #pragma cetus reduction(+: tx[il])  */
				for (ije1=0; ije1<nnje; ije1 ++ )
				{
					/* the first column, col=0, is an edge of face iface. */
					/* the value on the three interior collocation points, tx, is */
					/* computed by applying mapping matrices qbnew to tmp. */
					/* the mapping result is divided by 2, because there will be */
					/* duplicated contribution from another face sharing this edge. */
					col=0;
					#pragma cetus private(i, il, j) 
					#pragma loop name transf#0#0#1#0 
					/* #pragma cetus reduction(+: tx[il])  */
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][i][col];
						#pragma cetus private(j) 
						#pragma loop name transf#0#0#1#0#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (j=0; j<5; j ++ )
						{
							tx[il]=(tx[il]+((qbnew[ije1][j][i-1]*tmp[ije1][j][col])*0.5));
						}
					}
					/* for column 1 ~ lx-2 */
					#pragma cetus private(col, i, il, j) 
					#pragma loop name transf#0#0#1#1 
					/* #pragma cetus reduction(+: tx[il])  */
					for (col=1; col<(5-1); col ++ )
					{
						/* when i=0 or LX1-1, the collocation points are also on an edge of */
						/* the face, so the mapping result also needs to be divided by 2 */
						i=v_end[ije1];
						il=idel[ie][iface][i][col];
						tx[il]=(tx[il]+(tmp[ije1][i][col]*0.5));
						/* compute the value at interior collocation points in */
						/* columns 1 ~ LX1-1 */
						#pragma cetus private(i, il, j) 
						#pragma loop name transf#0#0#1#1#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (i=1; i<(5-1); i ++ )
						{
							il=idel[ie][iface][i][col];
							#pragma cetus private(j) 
							#pragma loop name transf#0#0#1#1#0#0 
							/* #pragma cetus reduction(+: tx[il])  */
							for (j=0; j<5; j ++ )
							{
								tx[il]=(tx[il]+(qbnew[ije1][j][i-1]*tmp[ije1][j][col]));
							}
						}
					}
					/* same as col=0 */
					col=(5-1);
					#pragma cetus private(i, il, j) 
					#pragma loop name transf#0#0#1#2 
					/* #pragma cetus reduction(+: tx[il])  */
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][i][col];
						#pragma cetus private(j) 
						#pragma loop name transf#0#0#1#2#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (j=0; j<5; j ++ )
						{
							tx[il]=(tx[il]+((qbnew[ije1][j][i-1]*tmp[ije1][j][col])*0.5));
						}
					}
				}
				/* for conforming faces */
			}
			else
			{
				/* face interior */
				#pragma cetus private(col, i, ig, il) 
				#pragma loop name transf#0#0#2 
				for (col=1; col<(5-1); col ++ )
				{
					#pragma cetus private(i, ig, il) 
					#pragma loop name transf#0#0#2#0 
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][col][i];
						ig=idmo[ie][iface][0][0][col][i];
						tx[il]=tmor[ig];
					}
				}
				/* edges of conforming faces */
				/* if local edge 0 is a nonconforming edge */
				if (idmo[ie][iface][0][0][0][5-1]!=( - 1))
				{
					#pragma cetus private(i, ig, ije1, il, j) 
					#pragma loop name transf#0#0#3 
					/* #pragma cetus reduction(+: tx[il])  */
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][0][i];
						#pragma cetus private(ig, ije1, j) 
						#pragma loop name transf#0#0#3#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (ije1=0; ije1<2; ije1 ++ )
						{
							#pragma cetus private(ig, j) 
							#pragma loop name transf#0#0#3#0#0 
							/* #pragma cetus reduction(+: tx[il])  */
							for (j=0; j<5; j ++ )
							{
								ig=idmo[ie][iface][ije1][0][0][j];
								tx[il]=(tx[il]+((qbnew[ije1][j][i-1]*tmor[ig])*0.5));
							}
						}
					}
					/* if local edge 0 is a conforming edge */
				}
				else
				{
					#pragma cetus private(i, ig, il) 
					#pragma loop name transf#0#0#4 
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][0][i];
						ig=idmo[ie][iface][0][0][0][i];
						tx[il]=tmor[ig];
					}
				}
				/* if local edge 1 is a nonconforming edge */
				if (idmo[ie][iface][1][0][1][5-1]!=( - 1))
				{
					#pragma cetus private(i, ig, ije1, il, j) 
					#pragma loop name transf#0#0#5 
					/* #pragma cetus reduction(+: tx[il])  */
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][i][5-1];
						#pragma cetus private(ig, ije1, j) 
						#pragma loop name transf#0#0#5#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (ije1=0; ije1<2; ije1 ++ )
						{
							#pragma cetus private(ig, j) 
							#pragma loop name transf#0#0#5#0#0 
							/* #pragma cetus reduction(+: tx[il])  */
							for (j=0; j<5; j ++ )
							{
								ig=idmo[ie][iface][1][ije1][j][5-1];
								tx[il]=(tx[il]+((qbnew[ije1][j][i-1]*tmor[ig])*0.5));
							}
						}
					}
					/* if local edge 1 is a conforming edge */
				}
				else
				{
					#pragma cetus private(i, ig, il) 
					#pragma loop name transf#0#0#6 
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][i][5-1];
						ig=idmo[ie][iface][0][0][i][5-1];
						tx[il]=tmor[ig];
					}
				}
				/* if local edge 2 is a nonconforming edge */
				if (idmo[ie][iface][0][1][5-1][1]!=( - 1))
				{
					#pragma cetus private(i, ig, ije1, il, j) 
					#pragma loop name transf#0#0#7 
					/* #pragma cetus reduction(+: tx[il])  */
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][5-1][i];
						#pragma cetus private(ig, ije1, j) 
						#pragma loop name transf#0#0#7#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (ije1=0; ije1<2; ije1 ++ )
						{
							#pragma cetus private(ig, j) 
							#pragma loop name transf#0#0#7#0#0 
							/* #pragma cetus reduction(+: tx[il])  */
							for (j=0; j<5; j ++ )
							{
								ig=idmo[ie][iface][ije1][1][5-1][j];
								tx[il]=(tx[il]+((qbnew[ije1][j][i-1]*tmor[ig])*0.5));
							}
						}
					}
					/* if local edge 2 is a conforming edge */
				}
				else
				{
					#pragma cetus private(i, ig, il) 
					#pragma loop name transf#0#0#8 
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][5-1][i];
						ig=idmo[ie][iface][0][0][5-1][i];
						tx[il]=tmor[ig];
					}
				}
				/* if local edge 3 is a nonconforming edge */
				if (idmo[ie][iface][0][0][5-1][0]!=( - 1))
				{
					#pragma cetus private(i, ig, ije1, il, j) 
					#pragma loop name transf#0#0#9 
					/* #pragma cetus reduction(+: tx[il])  */
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][i][0];
						#pragma cetus private(ig, ije1, j) 
						#pragma loop name transf#0#0#9#0 
						/* #pragma cetus reduction(+: tx[il])  */
						for (ije1=0; ije1<2; ije1 ++ )
						{
							#pragma cetus private(ig, j) 
							#pragma loop name transf#0#0#9#0#0 
							/* #pragma cetus reduction(+: tx[il])  */
							for (j=0; j<5; j ++ )
							{
								ig=idmo[ie][iface][0][ije1][j][0];
								tx[il]=(tx[il]+((qbnew[ije1][j][i-1]*tmor[ig])*0.5));
							}
						}
					}
					/* if local edge 3 is a conforming edge */
				}
				else
				{
					#pragma cetus private(i, ig, il) 
					#pragma loop name transf#0#0#10 
					for (i=1; i<(5-1); i ++ )
					{
						il=idel[ie][iface][i][0];
						ig=idmo[ie][iface][0][0][i][0];
						tx[il]=tmor[ig];
					}
				}
			}
		}
	}
}

/* ------------------------------------------------------------------ */
/* Map from element(tx) to mortar(tmor). */
/* tmor sums contributions from all elements. */
/* ------------------------------------------------------------------ */
void transfb(double tmor[], double tx[])
{
	const double third = 1.0/3.0;
	int shift;
	double tmp, tmp1, temp[2][5][5], top[2][5];
	int il1, il2, il3, il4, ig1, ig2, ig3, ig4, ie, iface, nnje;
	int ije1, ije2, col, i, j, ije, ig, il;
	#pragma cetus private(ie) 
	#pragma loop name transfb#0 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(3L*nmor)))) private(ie)
	for (ie=0; ie<nmor; ie ++ )
	{
		tmor[ie]=0.0;
	}
	#pragma cetus private(col, i, ie, iface, ig, ig1, ig2, ig3, ig4, ije, ije1, ije2, il, il1, il2, il3, il4, j, nnje, shift, tmp, tmp1) 
	#pragma loop name transfb#1 
	/* #pragma cetus reduction(+: tmor[ig1], tmor[ig2], tmor[ig3], tmor[ig4], tmor[ig])  */
	for (ie=0; ie<nelt; ie ++ )
	{
		#pragma cetus private(col, i, iface, ig, ig1, ig2, ig3, ig4, ije, ije1, ije2, il, il1, il2, il3, il4, j, nnje, shift, tmp, tmp1) 
		#pragma loop name transfb#1#0 
		/* #pragma cetus reduction(+: tmor[ig1], tmor[ig2], tmor[ig3], tmor[ig4], tmor[ig])  */
		for (iface=0; iface<6; iface ++ )
		{
			/* nnje=1 for conforming faces, nnje=2 for nonconforming faces */
			if (cbc[ie][iface]==3)
			{
				nnje=2;
			}
			else
			{
				nnje=1;
			}
			/* get collocation point index of four local corners on the face */
			il1=idel[ie][iface][0][0];
			il2=idel[ie][iface][0][5-1];
			il3=idel[ie][iface][5-1][0];
			il4=idel[ie][iface][5-1][5-1];
			/* get the mortar indices of the four local corners */
			ig1=idmo[ie][iface][0][0][0][0];
			ig2=idmo[ie][iface][1][0][0][5-1];
			ig3=idmo[ie][iface][0][1][5-1][0];
			ig4=idmo[ie][iface][1][1][5-1][5-1];
			/* sum the values from tx to tmor for these four local corners */
			/* only 13 of the value is summed, since there will be two duplicated */
			/* contributions from the other two faces sharing this vertex */
			tmor[ig1]+=(tx[il1]*third);
			tmor[ig2]+=(tx[il2]*third);
			tmor[ig3]+=(tx[il3]*third);
			tmor[ig4]+=(tx[il4]*third);
			/* for nonconforming faces */
			if (nnje==2)
			{
				r_init((double * )temp, (5*5)*2, 0.0);
				/* nonconforming faces have four pieces of mortar, first map tx to */
				/* two intermediate mortars stored in temp */
				#pragma cetus private(col, i, ije2, il, j, shift, tmp) 
				#pragma loop name transfb#1#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for if((10000<(1L+(469L*nnje)))) private(col, i, ije2, il, j, shift, tmp)
				for (ije2=0; ije2<nnje; ije2 ++ )
				{
					shift=ije2;
					#pragma cetus private(col, i, il, j, tmp) 
					#pragma loop name transfb#1#0#0#0 
					for (col=0; col<5; col ++ )
					{
						/* For mortar points on face edge (top and bottom), copy the */
						/* value from tx to temp */
						il=idel[ie][iface][v_end[ije2]][col];
						temp[ije2][v_end[ije2]][col]=tx[il];
						/* For mortar points on face edge (top and bottom), calculate */
						/* the interior points' contribution to them, i.e. top() */
						j=v_end[ije2];
						tmp=0.0;
						#pragma cetus private(i, il) 
						#pragma loop name transfb#1#0#0#0#0 
						/* #pragma cetus reduction(+: tmp)  */
						for (i=1; i<(5-1); i ++ )
						{
							il=idel[ie][iface][i][col];
							tmp=(tmp+(qbnew[ije2][j][i-1]*tx[il]));
						}
						top[ije2][col]=tmp;
						/* Use mapping matrices qbnew to map the value from tx to temp */
						/* for mortar points not on the top bottom face edge. */
						#pragma cetus private(i, il, j, tmp) 
						#pragma loop name transfb#1#0#0#0#1 
						for (j=((2-shift)-1); j<(5-shift); j ++ )
						{
							tmp=0.0;
							#pragma cetus private(i, il) 
							#pragma loop name transfb#1#0#0#0#1#0 
							/* #pragma cetus reduction(+: tmp)  */
							for (i=1; i<(5-1); i ++ )
							{
								il=idel[ie][iface][i][col];
								tmp=(tmp+(qbnew[ije2][j][i-1]*tx[il]));
							}
							;
							temp[ije2][j][col]=(tmp+temp[ije2][j][col]);
						}
					}
				}
				/* mapping from temp to tmor */
				#pragma cetus private(col, i, ig, ije1, ije2, j, shift, tmp, tmp1) 
				#pragma loop name transfb#1#0#1 
				/* #pragma cetus reduction(+: tmor[ig])  */
				for (ije1=0; ije1<nnje; ije1 ++ )
				{
					shift=ije1;
					#pragma cetus private(col, i, ig, ije2, j, tmp, tmp1) 
					#pragma loop name transfb#1#0#1#0 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (ije2=0; ije2<nnje; ije2 ++ )
					{
						/* for each column of collocation points on a piece of mortar */
						#pragma cetus private(col, i, ig, j, tmp) 
						#pragma loop name transfb#1#0#1#0#0 
						/* #pragma cetus reduction(+: tmor[ig])  */
						for (col=((2-shift)-1); col<(5-shift); col ++ )
						{
							/* For the end point, which is on an edge (local edge 1,3), */
							/* the contribution is halved since there will be duplicated */
							/* contribution from another face sharing this edge. */
							ig=idmo[ie][iface][ije2][ije1][col][v_end[ije2]];
							tmor[ig]+=(temp[ije1][col][v_end[ije2]]*0.5);
							/* In each row of collocation points on a piece of mortar, */
							/* sum the contributions from interior collocation points */
							/* (i=1,LX1-2) */
							#pragma cetus private(i, ig, j, tmp) 
							#pragma loop name transfb#1#0#1#0#0#0 
							/* #pragma cetus reduction(+: tmor[ig])  */
							for (j=0; j<5; j ++ )
							{
								tmp=0.0;
								#pragma cetus private(i) 
								#pragma loop name transfb#1#0#1#0#0#0#0 
								/* #pragma cetus reduction(+: tmp)  */
								for (i=1; i<(5-1); i ++ )
								{
									tmp=(tmp+(qbnew[ije2][j][i-1]*temp[ije1][col][i]));
								}
								ig=idmo[ie][iface][ije2][ije1][col][j];
								tmor[ig]+=tmp;
							}
						}
						/* For tmor on local edge 0 and 2, tmp is the contribution from */
						/* an edge, so it is halved because of duplicated contribution */
						/* from another face sharing this edge. tmp1 is contribution */
						/* from face interior. */
						col=v_end[ije1];
						ig=idmo[ie][iface][ije2][ije1][col][v_end[ije2]];
						tmor[ig]+=(top[ije1][v_end[ije2]]*0.5);
						#pragma cetus private(i, ig, j, tmp, tmp1) 
						#pragma loop name transfb#1#0#1#0#1 
						/* #pragma cetus reduction(+: tmor[ig])  */
						for (j=0; j<5; j ++ )
						{
							tmp=0.0;
							tmp1=0.0;
							#pragma cetus private(i) 
							#pragma loop name transfb#1#0#1#0#1#0 
							/* #pragma cetus reduction(+: tmp, tmp1)  */
							for (i=1; i<(5-1); i ++ )
							{
								tmp=(tmp+(qbnew[ije2][j][i-1]*temp[ije1][col][i]));
								tmp1=(tmp1+(qbnew[ije2][j][i-1]*top[ije1][i]));
							}
							ig=idmo[ie][iface][ije2][ije1][col][j];
							tmor[ig]+=((tmp*0.5)+tmp1);
						}
					}
				}
				/* for conforming faces */
			}
			else
			{
				/* face interior */
				#pragma cetus private(col, ig, il, j) 
				#pragma loop name transfb#1#0#2 
				/* #pragma cetus reduction(+: tmor[ig])  */
				for (col=1; col<(5-1); col ++ )
				{
					#pragma cetus private(ig, il, j) 
					#pragma loop name transfb#1#0#2#0 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (j=1; j<(5-1); j ++ )
					{
						il=idel[ie][iface][col][j];
						ig=idmo[ie][iface][0][0][col][j];
						tmor[ig]+=tx[il];
					}
				}
				/* edges of conforming faces */
				/* if local edge 0 is a nonconforming edge */
				if (idmo[ie][iface][0][0][0][5-1]!=( - 1))
				{
					#pragma cetus private(i, ig, ije, il, j, tmp) 
					#pragma loop name transfb#1#0#3 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (ije=0; ije<2; ije ++ )
					{
						#pragma cetus private(i, ig, il, j, tmp) 
						#pragma loop name transfb#1#0#3#0 
						/* #pragma cetus reduction(+: tmor[ig])  */
						for (j=0; j<5; j ++ )
						{
							tmp=0.0;
							#pragma cetus private(i, il) 
							#pragma loop name transfb#1#0#3#0#0 
							/* #pragma cetus reduction(+: tmp)  */
							for (i=1; i<(5-1); i ++ )
							{
								il=idel[ie][iface][0][i];
								tmp=(tmp+(qbnew[ije][j][i-1]*tx[il]));
							}
							ig=idmo[ie][iface][ije][0][0][j];
							tmor[ig]+=(tmp*0.5);
						}
					}
					/* if local edge 0 is a conforming edge */
				}
				else
				{
					#pragma cetus private(ig, il, j) 
					#pragma loop name transfb#1#0#4 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (j=1; j<(5-1); j ++ )
					{
						il=idel[ie][iface][0][j];
						ig=idmo[ie][iface][0][0][0][j];
						tmor[ig]+=(tx[il]*0.5);
					}
				}
				/* if local edge 1 is a nonconforming edge */
				if (idmo[ie][iface][1][0][1][5-1]!=( - 1))
				{
					#pragma cetus private(i, ig, ije, il, j, tmp) 
					#pragma loop name transfb#1#0#5 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (ije=0; ije<2; ije ++ )
					{
						#pragma cetus private(i, ig, il, j, tmp) 
						#pragma loop name transfb#1#0#5#0 
						/* #pragma cetus reduction(+: tmor[ig])  */
						for (j=0; j<5; j ++ )
						{
							tmp=0.0;
							#pragma cetus private(i, il) 
							#pragma loop name transfb#1#0#5#0#0 
							/* #pragma cetus reduction(+: tmp)  */
							for (i=1; i<(5-1); i ++ )
							{
								il=idel[ie][iface][i][5-1];
								tmp=(tmp+(qbnew[ije][j][i-1]*tx[il]));
							}
							ig=idmo[ie][iface][1][ije][j][5-1];
							tmor[ig]+=(tmp*0.5);
						}
					}
					/* if local edge 1 is a conforming edge */
				}
				else
				{
					#pragma cetus private(ig, il, j) 
					#pragma loop name transfb#1#0#6 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (j=1; j<(5-1); j ++ )
					{
						il=idel[ie][iface][j][5-1];
						ig=idmo[ie][iface][0][0][j][5-1];
						tmor[ig]+=(tx[il]*0.5);
					}
				}
				/* if local edge 2 is a nonconforming edge */
				if (idmo[ie][iface][0][1][5-1][1]!=( - 1))
				{
					#pragma cetus private(i, ig, ije, il, j, tmp) 
					#pragma loop name transfb#1#0#7 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (ije=0; ije<2; ije ++ )
					{
						#pragma cetus private(i, ig, il, j, tmp) 
						#pragma loop name transfb#1#0#7#0 
						/* #pragma cetus reduction(+: tmor[ig])  */
						for (j=0; j<5; j ++ )
						{
							tmp=0.0;
							#pragma cetus private(i, il) 
							#pragma loop name transfb#1#0#7#0#0 
							/* #pragma cetus reduction(+: tmp)  */
							for (i=1; i<(5-1); i ++ )
							{
								il=idel[ie][iface][5-1][i];
								tmp=(tmp+(qbnew[ije][j][i-1]*tx[il]));
							}
							ig=idmo[ie][iface][ije][1][5-1][j];
							tmor[ig]+=(tmp*0.5);
						}
					}
					/* if local edge 2 is a conforming edge */
				}
				else
				{
					#pragma cetus private(ig, il, j) 
					#pragma loop name transfb#1#0#8 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (j=1; j<(5-1); j ++ )
					{
						il=idel[ie][iface][5-1][j];
						ig=idmo[ie][iface][0][0][5-1][j];
						tmor[ig]+=(tx[il]*0.5);
					}
				}
				/* if local edge 3 is a nonconforming edge */
				if (idmo[ie][iface][0][0][5-1][0]!=( - 1))
				{
					#pragma cetus private(i, ig, ije, il, j, tmp) 
					#pragma loop name transfb#1#0#9 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (ije=0; ije<2; ije ++ )
					{
						#pragma cetus private(i, ig, il, j, tmp) 
						#pragma loop name transfb#1#0#9#0 
						/* #pragma cetus reduction(+: tmor[ig])  */
						for (j=0; j<5; j ++ )
						{
							tmp=0.0;
							#pragma cetus private(i, il) 
							#pragma loop name transfb#1#0#9#0#0 
							/* #pragma cetus reduction(+: tmp)  */
							for (i=1; i<(5-1); i ++ )
							{
								il=idel[ie][iface][i][0];
								tmp=(tmp+(qbnew[ije][j][i-1]*tx[il]));
							}
							ig=idmo[ie][iface][0][ije][j][0];
							tmor[ig]+=(tmp*0.5);
						}
					}
					/* if local edge 3 is a conforming edge */
				}
				else
				{
					#pragma cetus private(ig, il, j) 
					#pragma loop name transfb#1#0#10 
					/* #pragma cetus reduction(+: tmor[ig])  */
					for (j=1; j<(5-1); j ++ )
					{
						il=idel[ie][iface][j][0];
						ig=idmo[ie][iface][0][0][j][0];
						tmor[ig]+=(tx[il]*0.5);
					}
				}
			}
		}
	}
}

/* -------------------------------------------------------------- */
/* This subroutine performs the edge to mortar mapping and */
/* calculates the mapping result on the mortar point at a vertex */
/* under situation 1,2, or 3. */
/* n refers to the configuration of three edges sharing a vertex, */
/* n = 1: only one edge is nonconforming */
/* n = 2: two edges are nonconforming */
/* n = 3: three edges are nonconforming */
/* ------------------------------------------------------------------- */
void transfb_cor_e(int n, double * tmor, double tx[5][5][5])
{
	double tmp;
	int i;
	tmp=tx[0][0][0];
	#pragma cetus private(i) 
	#pragma loop name transfb_cor_e#0 
	#pragma cetus reduction(+: tmp) 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
	*/
	for (i=1; i<(5-1); i ++ )
	{
		tmp=(tmp+(qbnew[0][0][i-1]*tx[0][0][i]));
	}
	if (n>1)
	{
		#pragma cetus private(i) 
		#pragma loop name transfb_cor_e#1 
		#pragma cetus reduction(+: tmp) 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
		*/
		for (i=1; i<(5-1); i ++ )
		{
			tmp=(tmp+(qbnew[0][0][i-1]*tx[0][i][0]));
		}
	}
	if (n==3)
	{
		#pragma cetus private(i) 
		#pragma loop name transfb_cor_e#2 
		#pragma cetus reduction(+: tmp) 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
		*/
		for (i=1; i<(5-1); i ++ )
		{
			tmp=(tmp+(qbnew[0][0][i-1]*tx[i][0][0]));
		}
	}
	( * tmor)=tmp;
}

/* -------------------------------------------------------------- */
/* This subroutine performs the mapping from face to mortar. */
/* Output tmor is the mapping result on a mortar vertex */
/* of situations of three edges and three faces sharing a vertex: */
/* n=4: only one face is nonconforming */
/* n=5: one face and one edge are nonconforming */
/* n=6: two faces are nonconforming */
/* n=7: three faces are nonconforming */
/* -------------------------------------------------------------- */
void transfb_cor_f(int n, double * tmor, double tx[5][5][5])
{
	double temp[5], tmp;
	int col, i;
	r_init(temp, 5, 0.0);
	#pragma cetus private(col, i) 
	#pragma loop name transfb_cor_f#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i)
	*/
	for (col=0; col<5; col ++ )
	{
		temp[col]=tx[0][0][col];
		#pragma cetus private(i) 
		#pragma loop name transfb_cor_f#0#0 
		/* #pragma cetus reduction(+: temp[col])  */
		for (i=1; i<(5-1); i ++ )
		{
			temp[col]=(temp[col]+(qbnew[0][0][i-1]*tx[0][i][col]));
		}
	}
	tmp=temp[0];
	#pragma cetus private(i) 
	#pragma loop name transfb_cor_f#1 
	#pragma cetus reduction(+: tmp) 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
	*/
	for (i=1; i<(5-1); i ++ )
	{
		tmp=(tmp+(qbnew[0][0][i-1]*temp[i]));
	}
	if (n==5)
	{
		#pragma cetus private(i) 
		#pragma loop name transfb_cor_f#2 
		#pragma cetus reduction(+: tmp) 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
		*/
		for (i=1; i<(5-1); i ++ )
		{
			tmp=(tmp+(qbnew[0][0][i-1]*tx[i][0][0]));
		}
	}
	if (n>=6)
	{
		r_init(temp, 5, 0.0);
		#pragma cetus private(col, i) 
		#pragma loop name transfb_cor_f#3 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(col, i)
		*/
		for (col=0; col<5; col ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name transfb_cor_f#3#0 
			/* #pragma cetus reduction(+: temp[col])  */
			for (i=1; i<(5-1); i ++ )
			{
				temp[col]=(temp[col]+(qbnew[0][0][i-1]*tx[i][0][col]));
			}
		}
		tmp=(tmp+temp[0]);
		#pragma cetus private(i) 
		#pragma loop name transfb_cor_f#4 
		#pragma cetus reduction(+: tmp) 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
		*/
		for (i=1; i<(5-1); i ++ )
		{
			tmp=(tmp+(qbnew[0][0][i-1]*temp[i]));
		}
	}
	if (n==7)
	{
		r_init(temp, 5, 0.0);
		#pragma cetus private(col, i) 
		#pragma loop name transfb_cor_f#5 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(col, i)
		*/
		for (col=1; col<(5-1); col ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name transfb_cor_f#5#0 
			/* #pragma cetus reduction(+: temp[col])  */
			for (i=1; i<(5-1); i ++ )
			{
				temp[col]=(temp[col]+(qbnew[0][0][i-1]*tx[i][col][0]));
			}
		}
		#pragma cetus private(i) 
		#pragma loop name transfb_cor_f#6 
		#pragma cetus reduction(+: tmp) 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i) reduction(+: tmp)
		*/
		for (i=1; i<(5-1); i ++ )
		{
			tmp=(tmp+(qbnew[0][0][i-1]*temp[i]));
		}
	}
	( * tmor)=tmp;
}

/* ------------------------------------------------------------------------ */
/* Perform mortar to element mapping on a nonconforming face. */
/* This subroutin is used when all entries in tmor are zero except */
/* one tmor[j][i]=1. So this routine is simplified. Only one piece of */
/* mortar  (tmor only has two indices) and one piece of intermediate */
/* mortar (tmp) are involved. */
/* ------------------------------------------------------------------------ */
void transf_nc(double tmor[5][5], double tx[5][5])
{
	double tmp[5][5];
	int col, i, j;
	r_init((double * )tmp, 5*5, 0.0);
	#pragma cetus private(col, i, j) 
	#pragma loop name transf_nc#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i, j)
	*/
	for (col=0; col<5; col ++ )
	{
		i=0;
		tmp[col][i]=tmor[col][i];
		#pragma cetus private(i, j) 
		#pragma loop name transf_nc#0#0 
		for (i=1; i<(5-1); i ++ )
		{
			#pragma cetus private(j) 
			#pragma loop name transf_nc#0#0#0 
			/* #pragma cetus reduction(+: tmp[col][i])  */
			for (j=0; j<5; j ++ )
			{
				tmp[col][i]=(tmp[col][i]+(qbnew[0][j][i-1]*tmor[col][j]));
			}
		}
	}
	#pragma cetus private(col, i, j) 
	#pragma loop name transf_nc#1 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i, j)
	*/
	for (col=0; col<5; col ++ )
	{
		i=0;
		tx[i][col]=(tx[i][col]+tmp[i][col]);
		#pragma cetus private(i, j) 
		#pragma loop name transf_nc#1#0 
		for (i=1; i<(5-1); i ++ )
		{
			#pragma cetus private(j) 
			#pragma loop name transf_nc#1#0#0 
			/* #pragma cetus reduction(+: tx[i][col])  */
			for (j=0; j<5; j ++ )
			{
				tx[i][col]=(tx[i][col]+(qbnew[0][j][i-1]*tmp[j][col]));
			}
		}
	}
}

/* ------------------------------------------------------------------------ */
/* Performs mapping from element to mortar when the nonconforming */
/* edges are shared by two conforming faces of an element. */
/* ------------------------------------------------------------------------ */
void transfb_nc0(double tmor[5][5], double tx[5][5][5])
{
	int i, j;
	r_init((double * )tmor, 5*5, 0.0);
	#pragma cetus private(i, j) 
	#pragma loop name transfb_nc0#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i, j)
	*/
	for (j=0; j<5; j ++ )
	{
		#pragma cetus private(i) 
		#pragma loop name transfb_nc0#0#0 
		/* #pragma cetus reduction(+: tmor[0][j])  */
		for (i=1; i<(5-1); i ++ )
		{
			tmor[0][j]=(tmor[0][j]+(qbnew[0][j][i-1]*tx[0][0][i]));
		}
	}
}

/* ------------------------------------------------------------------------ */
/* Maps values from element to mortar when the nonconforming edges are */
/* shared by two nonconforming faces of an element. */
/* Although each face shall have four pieces of mortar, only value in */
/* one piece (location (0,0)) is used in the calling routine so only */
/* the value in the first mortar is calculated in this subroutine. */
/* ------------------------------------------------------------------------ */
void transfb_nc2(double tmor[5][5], double tx[5][5])
{
	double bottom[5], temp[5][5];
	int col, j, i;
	r_init((double * )tmor, 5*5, 0.0);
	r_init((double * )temp, 5*5, 0.0);
	tmor[0][0]=tx[0][0];
	/* mapping from tx to intermediate mortar temp + bottom */
	#pragma cetus private(col, i, j) 
	#pragma loop name transfb_nc2#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i, j)
	*/
	for (col=0; col<5; col ++ )
	{
		temp[0][col]=tx[0][col];
		j=0;
		bottom[col]=0.0;
		;
		#pragma cetus private(i) 
		#pragma loop name transfb_nc2#0#0 
		/* #pragma cetus reduction(+: bottom[col])  */
		for (i=1; i<(5-1); i ++ )
		{
			bottom[col]=(bottom[col]+(qbnew[0][j][i-1]*tx[i][col]));
		}
		#pragma cetus private(i, j) 
		#pragma loop name transfb_nc2#0#1 
		for (j=1; j<5; j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name transfb_nc2#0#1#0 
			/* #pragma cetus reduction(+: temp[j][col])  */
			for (i=1; i<(5-1); i ++ )
			{
				temp[j][col]=(temp[j][col]+(qbnew[0][j][i-1]*tx[i][col]));
			}
		}
	}
	/* from intermediate mortar to mortar */
	/* On the nonconforming edge, temp is divided by 2 as there will be */
	/* a duplicate contribution from another face sharing this edge */
	col=0;
	#pragma cetus private(i, j) 
	#pragma loop name transfb_nc2#1 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i, j)
	*/
	for (j=0; j<5; j ++ )
	{
		#pragma cetus private(i) 
		#pragma loop name transfb_nc2#1#0 
		/* #pragma cetus reduction(+: tmor[col][j])  */
		for (i=1; i<(5-1); i ++ )
		{
			tmor[col][j]=((tmor[col][j]+(qbnew[0][j][i-1]*bottom[i]))+((qbnew[0][j][i-1]*temp[col][i])*0.5));
		}
	}
	#pragma cetus private(col, i, j) 
	#pragma loop name transfb_nc2#2 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i, j)
	*/
	for (col=1; col<5; col ++ )
	{
		tmor[col][0]=(tmor[col][0]+temp[col][0]);
		#pragma cetus private(i, j) 
		#pragma loop name transfb_nc2#2#0 
		for (j=0; j<5; j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name transfb_nc2#2#0#0 
			/* #pragma cetus reduction(+: tmor[col][j])  */
			for (i=1; i<(5-1); i ++ )
			{
				tmor[col][j]=(tmor[col][j]+(qbnew[0][j][i-1]*temp[col][i]));
			}
		}
	}
}

/* ------------------------------------------------------------------------ */
/* Maps values from element to mortar when the nonconforming edges are */
/* shared by a nonconforming face and a conforming face of an element */
/* ------------------------------------------------------------------------ */
void transfb_nc1(double tmor[5][5], double tx[5][5])
{
	double bottom[5], temp[5][5];
	int col, j, i;
	r_init((double * )tmor, 5*5, 0.0);
	r_init((double * )temp, 5*5, 0.0);
	tmor[0][0]=tx[0][0];
	/* Contribution from the nonconforming faces */
	/* Since the calling subroutine is only interested in the value on the */
	/* mortar (location (0,0)), only this piece of mortar is calculated. */
	#pragma cetus private(col, i, j) 
	#pragma loop name transfb_nc1#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i, j)
	*/
	for (col=0; col<5; col ++ )
	{
		temp[0][col]=tx[0][col];
		j=0;
		bottom[col]=0.0;
		#pragma cetus private(i) 
		#pragma loop name transfb_nc1#0#0 
		/* #pragma cetus reduction(+: bottom[col])  */
		for (i=1; i<(5-1); i ++ )
		{
			bottom[col]=(bottom[col]+(qbnew[0][j][i-1]*tx[i][col]));
		}
		#pragma cetus private(i, j) 
		#pragma loop name transfb_nc1#0#1 
		for (j=1; j<5; j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name transfb_nc1#0#1#0 
			/* #pragma cetus reduction(+: temp[j][col])  */
			for (i=1; i<(5-1); i ++ )
			{
				temp[j][col]=(temp[j][col]+(qbnew[0][j][i-1]*tx[i][col]));
			}
		}
	}
	col=0;
	tmor[col][0]=(tmor[col][0]+bottom[0]);
	#pragma cetus private(i, j) 
	#pragma loop name transfb_nc1#1 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i, j)
	*/
	for (j=0; j<5; j ++ )
	{
		#pragma cetus private(i) 
		#pragma loop name transfb_nc1#1#0 
		/* #pragma cetus reduction(+: tmor[col][j])  */
		for (i=1; i<(5-1); i ++ )
		{
			/* temp is not divided by 2 here. It includes the contribution */
			/* from the other conforming face. */
			tmor[col][j]=((tmor[col][j]+(qbnew[0][j][i-1]*bottom[i]))+(qbnew[0][j][i-1]*temp[col][i]));
		}
	}
	#pragma cetus private(col, i, j) 
	#pragma loop name transfb_nc1#2 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(col, i, j)
	*/
	for (col=1; col<5; col ++ )
	{
		tmor[col][0]=(tmor[col][0]+temp[col][0]);
		#pragma cetus private(i, j) 
		#pragma loop name transfb_nc1#2#0 
		for (j=0; j<5; j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name transfb_nc1#2#0#0 
			/* #pragma cetus reduction(+: tmor[col][j])  */
			for (i=1; i<(5-1); i ++ )
			{
				tmor[col][j]=(tmor[col][j]+(qbnew[0][j][i-1]*temp[col][i]));
			}
		}
	}
}

/* ------------------------------------------------------------------- */
/* Prepare initial guess for cg. All values from conforming */
/* boundary are copied and summed on tmor. */
/* ------------------------------------------------------------------- */
void transfb_c(double tx[])
{
	const double third = 1.0/3.0;
	int il1, il2, il3, il4, ig1, ig2, ig3, ig4, ie, iface, col, j, ig, il;
	#pragma cetus private(j) 
	#pragma loop name transfb_c#0 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(3L*nmor)))) private(j)
	for (j=0; j<nmor; j ++ )
	{
		tmort[j]=0.0;
	}
	#pragma cetus parallel 
	#pragma cetus private(col, ie, iface, ig, ig1, ig2, ig3, ig4, il, il1, il2, il3, il4, j) 
	#pragma omp parallel if((10000<(37863521L+(831L*nelt)))) private(col, ie, iface, ig, ig1, ig2, ig3, ig4, il, il1, il2, il3, il4, j)
	{
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_0;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_1;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_2;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_3;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_4;
		for (reduce_span_0=0; reduce_span_0<1262100; reduce_span_0 ++ )
		{
			reduce[reduce_span_0]=0;
		}
		for (reduce_span_1=0; reduce_span_1<1262100; reduce_span_1 ++ )
		{
			reduce[reduce_span_1]=0;
		}
		for (reduce_span_2=0; reduce_span_2<1262100; reduce_span_2 ++ )
		{
			reduce[reduce_span_2]=0;
		}
		for (reduce_span_3=0; reduce_span_3<1262100; reduce_span_3 ++ )
		{
			reduce[reduce_span_3]=0;
		}
		for (reduce_span_4=0; reduce_span_4<1262100; reduce_span_4 ++ )
		{
			reduce[reduce_span_4]=0;
		}
		#pragma loop name transfb_c#1 
		#pragma cetus for  
		#pragma omp for
		for (ie=0; ie<nelt; ie ++ )
		{
			#pragma cetus private(col, iface, ig, ig1, ig2, ig3, ig4, il, il1, il2, il3, il4, j) 
			#pragma loop name transfb_c#1#0 
			/* #pragma cetus reduction(+: tmort[ig1], tmort[ig2], tmort[ig3], tmort[ig4], tmort[ig])  */
			for (iface=0; iface<6; iface ++ )
			{
				if (cbc[ie][iface]!=3)
				{
					il1=idel[ie][iface][0][0];
					il2=idel[ie][iface][0][5-1];
					il3=idel[ie][iface][5-1][0];
					il4=idel[ie][iface][5-1][5-1];
					ig1=idmo[ie][iface][0][0][0][0];
					ig2=idmo[ie][iface][1][0][0][5-1];
					ig3=idmo[ie][iface][0][1][5-1][0];
					ig4=idmo[ie][iface][1][1][5-1][5-1];
					reduce[ig1]+=(tx[il1]*third);
					reduce[ig2]+=(tx[il2]*third);
					reduce[ig3]+=(tx[il3]*third);
					reduce[ig4]+=(tx[il4]*third);
					#pragma cetus private(col, ig, il, j) 
					#pragma loop name transfb_c#1#0#0 
					/* #pragma cetus reduction(+: tmort[ig])  */
					for (col=1; col<(5-1); col ++ )
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c#1#0#0#0 
						/* #pragma cetus reduction(+: tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][col][j];
							ig=idmo[ie][iface][0][0][col][j];
							reduce[ig]+=tx[il];
						}
					}
					if (idmo[ie][iface][0][0][0][5-1]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c#1#0#1 
						/* #pragma cetus reduction(+: tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][0][j];
							ig=idmo[ie][iface][0][0][0][j];
							reduce[ig]+=(tx[il]*0.5);
						}
					}
					if (idmo[ie][iface][1][0][1][5-1]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c#1#0#2 
						/* #pragma cetus reduction(+: tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][j][5-1];
							ig=idmo[ie][iface][0][0][j][5-1];
							reduce[ig]+=(tx[il]*0.5);
						}
					}
					if (idmo[ie][iface][0][1][5-1][1]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c#1#0#3 
						/* #pragma cetus reduction(+: tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][5-1][j];
							ig=idmo[ie][iface][0][0][5-1][j];
							reduce[ig]+=(tx[il]*0.5);
						}
					}
					if (idmo[ie][iface][0][0][5-1][0]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c#1#0#4 
						/* #pragma cetus reduction(+: tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][j][0];
							ig=idmo[ie][iface][0][0][j][0];
							reduce[ig]+=(tx[il]*0.5);
						}
					}
				}
			}
		}
		#pragma cetus critical  
		#pragma omp critical
		{
			for (reduce_span_0=0; reduce_span_0<1262100; reduce_span_0 ++ )
			{
				tmort[reduce_span_0]+=reduce[reduce_span_0];
			}
			for (reduce_span_1=0; reduce_span_1<1262100; reduce_span_1 ++ )
			{
				tmort[reduce_span_1]+=reduce[reduce_span_1];
			}
			for (reduce_span_2=0; reduce_span_2<1262100; reduce_span_2 ++ )
			{
				tmort[reduce_span_2]+=reduce[reduce_span_2];
			}
			for (reduce_span_3=0; reduce_span_3<1262100; reduce_span_3 ++ )
			{
				tmort[reduce_span_3]+=reduce[reduce_span_3];
			}
			for (reduce_span_4=0; reduce_span_4<1262100; reduce_span_4 ++ )
			{
				tmort[reduce_span_4]+=reduce[reduce_span_4];
			}
		}
	}
}

/* ------------------------------------------------------------------- */
/* Prepare initial guess for CG. All values from conforming */
/* boundary are copied and summed in tmort. */
/* mormult is multiplicity, which is used to average tmort. */
/* ------------------------------------------------------------------- */
void transfb_c_2(double tx[])
{
	const double third = 1.0/3.0;
	int il1, il2, il3, il4, ig1, ig2, ig3, ig4, ie, iface, col, j, ig, il;
	#pragma cetus private(j) 
	#pragma loop name transfb_c_2#0 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(3L*nmor)))) private(j)
	for (j=0; j<nmor; j ++ )
	{
		tmort[j]=0.0;
	}
	#pragma cetus private(j) 
	#pragma loop name transfb_c_2#1 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(3L*nmor)))) private(j)
	for (j=0; j<nmor; j ++ )
	{
		mormult[j]=0.0;
	}
	#pragma cetus parallel 
	#pragma cetus private(col, ie, iface, ig, ig1, ig2, ig3, ig4, il, il1, il2, il3, il4, j) 
	#pragma omp parallel if((10000<(75727041L+(981L*nelt)))) private(col, ie, iface, ig, ig1, ig2, ig3, ig4, il, il1, il2, il3, il4, j)
	{
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_0;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_1;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_2;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_3;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_4;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_5;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_6;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_7;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_8;
		double * reduce = (double * )malloc(1262100*sizeof (double));
		int reduce_span_9;
		for (reduce_span_0=0; reduce_span_0<1262100; reduce_span_0 ++ )
		{
			reduce[reduce_span_0]=0;
		}
		for (reduce_span_1=0; reduce_span_1<1262100; reduce_span_1 ++ )
		{
			reduce[reduce_span_1]=0;
		}
		for (reduce_span_2=0; reduce_span_2<1262100; reduce_span_2 ++ )
		{
			reduce[reduce_span_2]=0;
		}
		for (reduce_span_3=0; reduce_span_3<1262100; reduce_span_3 ++ )
		{
			reduce[reduce_span_3]=0;
		}
		for (reduce_span_4=0; reduce_span_4<1262100; reduce_span_4 ++ )
		{
			reduce[reduce_span_4]=0;
		}
		for (reduce_span_5=0; reduce_span_5<1262100; reduce_span_5 ++ )
		{
			reduce[reduce_span_5]=0;
		}
		for (reduce_span_6=0; reduce_span_6<1262100; reduce_span_6 ++ )
		{
			reduce[reduce_span_6]=0;
		}
		for (reduce_span_7=0; reduce_span_7<1262100; reduce_span_7 ++ )
		{
			reduce[reduce_span_7]=0;
		}
		for (reduce_span_8=0; reduce_span_8<1262100; reduce_span_8 ++ )
		{
			reduce[reduce_span_8]=0;
		}
		for (reduce_span_9=0; reduce_span_9<1262100; reduce_span_9 ++ )
		{
			reduce[reduce_span_9]=0;
		}
		#pragma loop name transfb_c_2#2 
		#pragma cetus for  
		#pragma omp for
		for (ie=0; ie<nelt; ie ++ )
		{
			#pragma cetus private(col, iface, ig, ig1, ig2, ig3, ig4, il, il1, il2, il3, il4, j) 
			#pragma loop name transfb_c_2#2#0 
			/* #pragma cetus reduction(+: mormult[ig1], mormult[ig2], mormult[ig3], mormult[ig4], mormult[ig], tmort[ig1], tmort[ig2], tmort[ig3], tmort[ig4], tmort[ig])  */
			for (iface=0; iface<6; iface ++ )
			{
				if (cbc[ie][iface]!=3)
				{
					il1=idel[ie][iface][0][0];
					il2=idel[ie][iface][0][5-1];
					il3=idel[ie][iface][5-1][0];
					il4=idel[ie][iface][5-1][5-1];
					ig1=idmo[ie][iface][0][0][0][0];
					ig2=idmo[ie][iface][1][0][0][5-1];
					ig3=idmo[ie][iface][0][1][5-1][0];
					ig4=idmo[ie][iface][1][1][5-1][5-1];
					reduce[ig1]+=(tx[il1]*third);
					reduce[ig2]+=(tx[il2]*third);
					reduce[ig3]+=(tx[il3]*third);
					reduce[ig4]+=(tx[il4]*third);
					reduce[ig1]+=third;
					reduce[ig2]+=third;
					reduce[ig3]+=third;
					reduce[ig4]+=third;
					#pragma cetus private(col, ig, il, j) 
					#pragma loop name transfb_c_2#2#0#0 
					/* #pragma cetus reduction(+: mormult[ig], tmort[ig])  */
					for (col=1; col<(5-1); col ++ )
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c_2#2#0#0#0 
						/* #pragma cetus reduction(+: mormult[ig], tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][col][j];
							ig=idmo[ie][iface][0][0][col][j];
							reduce[ig]+=tx[il];
							reduce[ig]+=1.0;
						}
					}
					if (idmo[ie][iface][0][0][0][5-1]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c_2#2#0#1 
						/* #pragma cetus reduction(+: mormult[ig], tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][0][j];
							ig=idmo[ie][iface][0][0][0][j];
							reduce[ig]+=(tx[il]*0.5);
							reduce[ig]+=0.5;
						}
					}
					if (idmo[ie][iface][1][0][1][5-1]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c_2#2#0#2 
						/* #pragma cetus reduction(+: mormult[ig], tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][j][5-1];
							ig=idmo[ie][iface][0][0][j][5-1];
							reduce[ig]+=(tx[il]*0.5);
							reduce[ig]+=0.5;
						}
					}
					if (idmo[ie][iface][0][1][5-1][1]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c_2#2#0#3 
						/* #pragma cetus reduction(+: mormult[ig], tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][5-1][j];
							ig=idmo[ie][iface][0][0][5-1][j];
							reduce[ig]+=(tx[il]*0.5);
							reduce[ig]+=0.5;
						}
					}
					if (idmo[ie][iface][0][0][5-1][0]==( - 1))
					{
						#pragma cetus private(ig, il, j) 
						#pragma loop name transfb_c_2#2#0#4 
						/* #pragma cetus reduction(+: mormult[ig], tmort[ig])  */
						for (j=1; j<(5-1); j ++ )
						{
							il=idel[ie][iface][j][0];
							ig=idmo[ie][iface][0][0][j][0];
							reduce[ig]+=(tx[il]*0.5);
							reduce[ig]+=0.5;
						}
					}
				}
			}
		}
		#pragma cetus critical  
		#pragma omp critical
		{
			for (reduce_span_0=0; reduce_span_0<1262100; reduce_span_0 ++ )
			{
				mormult[reduce_span_0]+=reduce[reduce_span_0];
			}
			for (reduce_span_1=0; reduce_span_1<1262100; reduce_span_1 ++ )
			{
				tmort[reduce_span_1]+=reduce[reduce_span_1];
			}
			for (reduce_span_2=0; reduce_span_2<1262100; reduce_span_2 ++ )
			{
				mormult[reduce_span_2]+=reduce[reduce_span_2];
			}
			for (reduce_span_3=0; reduce_span_3<1262100; reduce_span_3 ++ )
			{
				tmort[reduce_span_3]+=reduce[reduce_span_3];
			}
			for (reduce_span_4=0; reduce_span_4<1262100; reduce_span_4 ++ )
			{
				tmort[reduce_span_4]+=reduce[reduce_span_4];
			}
			for (reduce_span_5=0; reduce_span_5<1262100; reduce_span_5 ++ )
			{
				tmort[reduce_span_5]+=reduce[reduce_span_5];
			}
			for (reduce_span_6=0; reduce_span_6<1262100; reduce_span_6 ++ )
			{
				tmort[reduce_span_6]+=reduce[reduce_span_6];
			}
			for (reduce_span_7=0; reduce_span_7<1262100; reduce_span_7 ++ )
			{
				mormult[reduce_span_7]+=reduce[reduce_span_7];
			}
			for (reduce_span_8=0; reduce_span_8<1262100; reduce_span_8 ++ )
			{
				mormult[reduce_span_8]+=reduce[reduce_span_8];
			}
			for (reduce_span_9=0; reduce_span_9<1262100; reduce_span_9 ++ )
			{
				mormult[reduce_span_9]+=reduce[reduce_span_9];
			}
		}
	}
}
