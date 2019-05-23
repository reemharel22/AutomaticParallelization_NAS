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
#include <math.h>
#include "header.h"
#include "timers.h"
/* --------------------------------------------------------- */
/* Advance the convection term using 4th order RK */
/* 1.ta1 is solution from last time step  */
/* 2.the heat source is considered part of ddx */
/* 3.trhs is right hand side for the diffusion equation */
/* 4.tmor is solution on mortar points, which will be used */
/*   as the initial guess when advancing the diffusion term  */
/* --------------------------------------------------------- */
void convect(logical ifmortar)
{
	double alpha2, tempa[5][5][5], rdtime, pidivalpha;
	double dtx1, dtx2, dtx3, src, rk1[5][5][5];
	double rk2[5][5][5], rk3[5][5][5], rk4[5][5][5];
	double temp[5][5][5], subtime[3], xx0[3], yy0[3], zz0[3];
	double dtime2, r2, sum, xloc[5], yloc[5], zloc[5];
	int k, iel, i, j, iside, isize, substep, ip;
	const double sixth = 1.0/6.0;
	if (timeron)
	{
		timer_start(3);
	}
	pidivalpha=(acos( - 1.0)/alpha);
	alpha2=(alpha*alpha);
	dtime2=(dtime/2.0);
	rdtime=(1.0/dtime);
	subtime[0]=time;
	subtime[1]=(time+dtime2);
	subtime[2]=(time+dtime);
	#pragma cetus private(substep) 
	#pragma loop name convect#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(substep)
	*/
	for (substep=0; substep<3; substep ++ )
	{
		xx0[substep]=((3.0/7.0)+(3.0*subtime[substep]));
		yy0[substep]=((2.0/7.0)+(3.0*subtime[substep]));
		zz0[substep]=((2.0/7.0)+(3.0*subtime[substep]));
	}
	#pragma cetus private(dtx1, dtx2, dtx3, i, iel, ip, iside, isize, j, k, r2, rk1, rk2, rk3, rk4, src, sum, temp, xloc, yloc, zloc) 
	#pragma loop name convect#1 
	for (iel=0; iel<nelt; iel ++ )
	{
		isize=size_e[iel];
		/*
		
		    xloc[i] is the location of i'th collocation in x direction in an element.
		    yloc[i] is the location of j'th collocation in y direction in an element.
		    zloc[i] is the location of k'th collocation in z direction in an element.
		   
		*/
		#pragma cetus private(i) 
		#pragma loop name convect#1#0 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i)
		*/
		for (i=0; i<5; i ++ )
		{
			xloc[i]=((xfrac[i]*(xc[iel][1]-xc[iel][0]))+xc[iel][0]);
		}
		#pragma cetus private(j) 
		#pragma loop name convect#1#1 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(j)
		*/
		for (j=0; j<5; j ++ )
		{
			yloc[j]=((xfrac[j]*(yc[iel][3]-yc[iel][0]))+yc[iel][0]);
		}
		#pragma cetus private(k) 
		#pragma loop name convect#1#2 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(k)
		*/
		for (k=0; k<5; k ++ )
		{
			zloc[k]=((xfrac[k]*(zc[iel][4]-zc[iel][0]))+zc[iel][0]);
		}
		#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum) 
		#pragma loop name convect#1#3 
		#pragma cetus parallel 
		#pragma omp parallel for private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum)
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, r2, src, sum) 
			#pragma loop name convect#1#3#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(dtx1, dtx2, dtx3, i, ip, r2, src, sum) 
				#pragma loop name convect#1#3#0#0 
				for (i=0; i<5; i ++ )
				{
					r2=((pow(xloc[i]-xx0[0], 2.0)+pow(yloc[j]-yy0[0], 2.0))+pow(zloc[k]-zz0[0], 2.0));
					if (r2<=alpha2)
					{
						src=(cos(sqrt(r2)*pidivalpha)+1.0);
					}
					else
					{
						src=0.0;
					}
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#3#0#0#0 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][i]*ta1[iel][k][j][ip]));
					}
					dtx1=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#3#0#0#1 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][j]*ta1[iel][k][ip][i]));
					}
					dtx2=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#3#0#0#2 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][k]*ta1[iel][ip][j][i]));
					}
					dtx3=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					rk1[k][j][i]=(((dtx1+dtx2)+dtx3)+src);
					temp[k][j][i]=(ta1[iel][k][j][i]+(dtime2*rk1[k][j][i]));
				}
			}
		}
		#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum) 
		#pragma loop name convect#1#4 
		#pragma cetus parallel 
		#pragma omp parallel for private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum)
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, r2, src, sum) 
			#pragma loop name convect#1#4#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(dtx1, dtx2, dtx3, i, ip, r2, src, sum) 
				#pragma loop name convect#1#4#0#0 
				for (i=0; i<5; i ++ )
				{
					r2=((pow(xloc[i]-xx0[1], 2.0)+pow(yloc[j]-yy0[1], 2.0))+pow(zloc[k]-zz0[1], 2.0));
					if (r2<=alpha2)
					{
						src=(cos(sqrt(r2)*pidivalpha)+1.0);
					}
					else
					{
						src=0.0;
					}
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#4#0#0#0 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][i]*temp[k][j][ip]));
					}
					dtx1=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#4#0#0#1 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][j]*temp[k][ip][i]));
					}
					dtx2=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#4#0#0#2 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][k]*temp[ip][j][i]));
					}
					dtx3=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					rk2[k][j][i]=(((dtx1+dtx2)+dtx3)+src);
					tempa[k][j][i]=(ta1[iel][k][j][i]+(dtime2*rk2[k][j][i]));
				}
			}
		}
		#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum) 
		#pragma loop name convect#1#5 
		#pragma cetus parallel 
		#pragma omp parallel for private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum)
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, r2, src, sum) 
			#pragma loop name convect#1#5#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(dtx1, dtx2, dtx3, i, ip, r2, src, sum) 
				#pragma loop name convect#1#5#0#0 
				for (i=0; i<5; i ++ )
				{
					r2=((pow(xloc[i]-xx0[1], 2.0)+pow(yloc[j]-yy0[1], 2.0))+pow(zloc[k]-zz0[1], 2.0));
					if (r2<=alpha2)
					{
						src=(cos(sqrt(r2)*pidivalpha)+1.0);
					}
					else
					{
						src=0.0;
					}
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#5#0#0#0 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][i]*tempa[k][j][ip]));
					}
					dtx1=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#5#0#0#1 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][j]*tempa[k][ip][i]));
					}
					dtx2=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#5#0#0#2 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][k]*tempa[ip][j][i]));
					}
					dtx3=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					rk3[k][j][i]=(((dtx1+dtx2)+dtx3)+src);
					temp[k][j][i]=(ta1[iel][k][j][i]+(dtime*rk3[k][j][i]));
				}
			}
		}
		#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum) 
		#pragma loop name convect#1#6 
		#pragma cetus parallel 
		#pragma omp parallel for private(dtx1, dtx2, dtx3, i, ip, j, k, r2, src, sum)
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(dtx1, dtx2, dtx3, i, ip, j, r2, src, sum) 
			#pragma loop name convect#1#6#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(dtx1, dtx2, dtx3, i, ip, r2, src, sum) 
				#pragma loop name convect#1#6#0#0 
				for (i=0; i<5; i ++ )
				{
					r2=((pow(xloc[i]-xx0[2], 2.0)+pow(yloc[j]-yy0[2], 2.0))+pow(zloc[k]-zz0[2], 2.0));
					if (r2<=alpha2)
					{
						src=(cos(sqrt(r2)*pidivalpha)+1.0);
					}
					else
					{
						src=0.0;
					}
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#6#0#0#0 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][i]*temp[k][j][ip]));
					}
					dtx1=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#6#0#0#1 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][j]*temp[k][ip][i]));
					}
					dtx2=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					sum=0.0;
					#pragma cetus private(ip) 
					#pragma loop name convect#1#6#0#0#2 
					/* #pragma cetus reduction(+: sum)  */
					for (ip=0; ip<5; ip ++ )
					{
						sum=(sum+(dxm1[ip][k]*temp[ip][j][i]));
					}
					dtx3=((( - 3.0)*sum)*xrm1_s[isize][k][j][i]);
					rk4[k][j][i]=(((dtx1+dtx2)+dtx3)+src);
					tempa[k][j][i]=(sixth*(((rk1[k][j][i]+(2.0*rk2[k][j][i]))+(2.0*rk3[k][j][i]))+rk4[k][j][i]));
				}
			}
		}
		/* apply boundary condition */
		#pragma cetus private(iside) 
		#pragma loop name convect#1#7 
		for (iside=0; iside<6; iside ++ )
		{
			if (cbc[iel][iside]==0)
			{
				facev(tempa, iside, 0.0);
			}
		}
		#pragma cetus private(i, j, k) 
		#pragma loop name convect#1#8 
		#pragma cetus parallel 
		/*
		Disabled due to low profitability: #pragma omp parallel for private(i, j, k)
		*/
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(i, j) 
			#pragma loop name convect#1#8#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(i) 
				#pragma loop name convect#1#8#0#0 
				for (i=0; i<5; i ++ )
				{
					trhs[iel][k][j][i]=(bm1_s[isize][k][j][i]*((ta1[iel][k][j][i]*rdtime)+tempa[k][j][i]));
					ta1[iel][k][j][i]=(ta1[iel][k][j][i]+(tempa[k][j][i]*dtime));
				}
			}
		}
	}
	/* get mortar for intial guess for CG */
	if (timeron)
	{
		timer_start(4);
	}
	if (ifmortar)
	{
		transfb_c_2((double * )ta1);
	}
	else
	{
		transfb_c((double * )ta1);
	}
	if (timeron)
	{
		timer_stop(4);
	}
	#pragma cetus private(i) 
	#pragma loop name convect#2 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(3L*nmor)))) private(i)
	for (i=0; i<nmor; i ++ )
	{
		tmort[i]=(tmort[i]/mormult[i]);
	}
	if (timeron)
	{
		timer_stop(3);
	}
}
