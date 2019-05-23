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
#include "timers.h"
/* --------------------------------------------------------------------- */
/* advance the diffusion term using CG iterations */
/* --------------------------------------------------------------------- */
void diffusion(logical ifmortar)
{
	double rho_aux, rho1, rho2, beta, cona;
	int iter, ie, im, iside, i, j, k;
	if (timeron)
	{
		timer_start(5);
	}
	/* set up diagonal preconditioner */
	if (ifmortar)
	{
		setuppc();
		setpcmo();
	}
	/* arrays t and umor are accumlators of (am pm) in the CG algorithm */
	/* (see the specification) */
	r_init_omp((double * )t, ntot, 0.0);
	#pragma cetus private(i) 
	#pragma loop name diffusion#0 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(3L*nmor)))) private(i)
	for (i=0; i<nmor; i ++ )
	{
		umor[i]=0.0;
	}
	/* calculate initial am (see specification) in CG algorithm */
	/* trhs and rmor are combined to generate r0 in CG algorithm. */
	/* pdiff and pmorx are combined to generate q0 in the CG algorithm. */
	/* rho1 is  (qm,rm) in the CG algorithm. */
	rho1=0.0;
	#pragma cetus private(i, ie, j, k) 
	#pragma loop name diffusion#1 
	#pragma cetus reduction(+: rho1) 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(593L*nelt)))) private(i, ie, j, k) reduction(+: rho1)
	for (ie=0; ie<nelt; ie ++ )
	{
		#pragma cetus private(i, j, k) 
		#pragma loop name diffusion#1#0 
		/* #pragma cetus reduction(+: rho1)  */
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(i, j) 
			#pragma loop name diffusion#1#0#0 
			/* #pragma cetus reduction(+: rho1)  */
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(i) 
				#pragma loop name diffusion#1#0#0#0 
				/* #pragma cetus reduction(+: rho1)  */
				for (i=0; i<5; i ++ )
				{
					pdiff[ie][k][j][i]=(dpcelm[ie][k][j][i]*trhs[ie][k][j][i]);
					rho1=(rho1+((trhs[ie][k][j][i]*pdiff[ie][k][j][i])*tmult[ie][k][j][i]));
				}
			}
		}
	}
	#pragma cetus private(im) 
	#pragma loop name diffusion#2 
	#pragma cetus reduction(+: rho1) 
	#pragma cetus parallel 
	#pragma omp parallel for if((10000<(1L+(4L*nmor)))) private(im) reduction(+: rho1)
	for (im=0; im<nmor; im ++ )
	{
		pmorx[im]=(dpcmor[im]*rmor[im]);
		rho1=(rho1+(rmor[im]*pmorx[im]));
	}
	/* ................................................................. */
	/* commence conjugate gradient iteration */
	/* ................................................................. */
	#pragma cetus private(beta, cona, i, ie, im, iside, iter, j, k, rho2, rho_aux) 
	#pragma loop name diffusion#3 
	for (iter=1; iter<=nmxh; iter ++ )
	{
		if (iter>1)
		{
			rho_aux=0.0;
			/* pdiffp and ppmor are combined to generate q_m+1 in the specification */
			/* rho_aux is (q_m+1,r_m+1) */
			#pragma cetus private(i, ie, j, k) 
			#pragma loop name diffusion#3#0 
			#pragma cetus reduction(+: rho_aux) 
			#pragma cetus parallel 
			#pragma omp parallel for if((10000<(1L+(593L*nelt)))) private(i, ie, j, k) reduction(+: rho_aux)
			for (ie=0; ie<nelt; ie ++ )
			{
				#pragma cetus private(i, j, k) 
				#pragma loop name diffusion#3#0#0 
				/* #pragma cetus reduction(+: rho_aux)  */
				for (k=0; k<5; k ++ )
				{
					#pragma cetus private(i, j) 
					#pragma loop name diffusion#3#0#0#0 
					/* #pragma cetus reduction(+: rho_aux)  */
					for (j=0; j<5; j ++ )
					{
						#pragma cetus private(i) 
						#pragma loop name diffusion#3#0#0#0#0 
						/* #pragma cetus reduction(+: rho_aux)  */
						for (i=0; i<5; i ++ )
						{
							pdiffp[ie][k][j][i]=(dpcelm[ie][k][j][i]*trhs[ie][k][j][i]);
							rho_aux=(rho_aux+((trhs[ie][k][j][i]*pdiffp[ie][k][j][i])*tmult[ie][k][j][i]));
						}
					}
				}
			}
			#pragma cetus private(im) 
			#pragma loop name diffusion#3#1 
			#pragma cetus reduction(+: rho_aux) 
			#pragma cetus parallel 
			#pragma omp parallel for if((10000<(1L+(4L*nmor)))) private(im) reduction(+: rho_aux)
			for (im=0; im<nmor; im ++ )
			{
				ppmor[im]=(dpcmor[im]*rmor[im]);
				rho_aux=(rho_aux+(rmor[im]*ppmor[im]));
			}
			/* compute bm (beta) in the specification */
			rho2=rho1;
			rho1=rho_aux;
			beta=(rho1/rho2);
			/* update p_m+1 in the specification */
			adds1m1((double * )pdiff, (double * )pdiffp, beta, ntot);
			adds1m1(pmorx, ppmor, beta, nmor);
		}
		/* compute matrix vector product: (theta pm) in the specification */
		if (timeron)
		{
			timer_start(6);
		}
		transf(pmorx, (double * )pdiff);
		if (timeron)
		{
			timer_stop(6);
		}
		/* compute pdiffp which is (A theta pm) in the specification */
		#pragma cetus private(ie) 
		#pragma loop name diffusion#3#2 
		for (ie=0; ie<nelt; ie ++ )
		{
			laplacian(pdiffp[ie], pdiff[ie], size_e[ie]);
		}
		/* compute ppmor which will be used to compute (thetaT A theta pm)  */
		/* in the specification */
		if (timeron)
		{
			timer_start(7);
		}
		transfb(ppmor, (double * )pdiffp);
		if (timeron)
		{
			timer_stop(7);
		}
		/* apply boundary condition */
		#pragma cetus private(ie, iside) 
		#pragma loop name diffusion#3#3 
		for (ie=0; ie<nelt; ie ++ )
		{
			#pragma cetus private(iside) 
			#pragma loop name diffusion#3#3#0 
			for (iside=0; iside<6; iside ++ )
			{
				if (cbc[ie][iside]==0)
				{
					facev(pdiffp[ie], iside, 0.0);
				}
			}
		}
		/* compute cona which is (pm,theta T A theta pm) */
		cona=0.0;
		#pragma cetus private(i, ie, j, k) 
		#pragma loop name diffusion#3#4 
		#pragma cetus reduction(+: cona) 
		#pragma cetus parallel 
		#pragma omp parallel for if((10000<(1L+(468L*nelt)))) private(i, ie, j, k) reduction(+: cona)
		for (ie=0; ie<nelt; ie ++ )
		{
			#pragma cetus private(i, j, k) 
			#pragma loop name diffusion#3#4#0 
			/* #pragma cetus reduction(+: cona)  */
			for (k=0; k<5; k ++ )
			{
				#pragma cetus private(i, j) 
				#pragma loop name diffusion#3#4#0#0 
				/* #pragma cetus reduction(+: cona)  */
				for (j=0; j<5; j ++ )
				{
					#pragma cetus private(i) 
					#pragma loop name diffusion#3#4#0#0#0 
					/* #pragma cetus reduction(+: cona)  */
					for (i=0; i<5; i ++ )
					{
						cona=(cona+((pdiff[ie][k][j][i]*pdiffp[ie][k][j][i])*tmult[ie][k][j][i]));
					}
				}
			}
		}
		#pragma cetus private(im) 
		#pragma loop name diffusion#3#5 
		#pragma cetus reduction(+: cona) 
		#pragma cetus parallel 
		#pragma omp parallel for if((10000<(1L+(4L*nmor)))) private(im) reduction(+: cona)
		for (im=0; im<nmor; im ++ )
		{
			ppmor[im]=(ppmor[im]*tmmor[im]);
			cona=(cona+(pmorx[im]*ppmor[im]));
		}
		/* compute am */
		cona=(rho1/cona);
		/* compute (am pm) */
		adds2m1((double * )t, (double * )pdiff, cona, ntot);
		adds2m1(umor, pmorx, cona, nmor);
		/* compute r_m+1 */
		adds2m1((double * )trhs, (double * )pdiffp,  - cona, ntot);
		adds2m1(rmor, ppmor,  - cona, nmor);
	}
	if (timeron)
	{
		timer_start(6);
	}
	transf(umor, (double * )t);
	if (timeron)
	{
		timer_stop(6);
	}
	if (timeron)
	{
		timer_stop(5);
	}
}

/* ------------------------------------------------------------------ */
/* compute  r = visc[A]x +[B]x on a given element. */
/* ------------------------------------------------------------------ */
void laplacian(double r[5][5][5], double u[5][5][5], int sizei)
{
	double rdtime;
	int i, j, k, iz;
	double tm1[5][5][5], tm2[5][5][5];
	rdtime=(1.0/dtime);
	r_init((double * )tm1, (5*5)*5, 0.0);
	#pragma cetus private(i, iz, j, k) 
	#pragma loop name laplacian#0 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i, iz, j, k)
	*/
	for (iz=0; iz<5; iz ++ )
	{
		#pragma cetus private(i, j, k) 
		#pragma loop name laplacian#0#0 
		/* #pragma cetus reduction(+: tm1[iz][j][i])  */
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(i, j) 
			#pragma loop name laplacian#0#0#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(i) 
				#pragma loop name laplacian#0#0#0#0 
				for (i=0; i<5; i ++ )
				{
					tm1[iz][j][i]=(tm1[iz][j][i]+(wdtdr[k][i]*u[iz][j][k]));
				}
			}
		}
	}
	r_init((double * )tm2, (5*5)*5, 0.0);
	#pragma cetus private(i, iz, j, k) 
	#pragma loop name laplacian#1 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i, iz, j, k)
	*/
	for (iz=0; iz<5; iz ++ )
	{
		#pragma cetus private(i, j, k) 
		#pragma loop name laplacian#1#0 
		/* #pragma cetus reduction(+: tm2[iz][j][i])  */
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(i, j) 
			#pragma loop name laplacian#1#0#0 
			for (j=0; j<5; j ++ )
			{
				#pragma cetus private(i) 
				#pragma loop name laplacian#1#0#0#0 
				for (i=0; i<5; i ++ )
				{
					tm2[iz][j][i]=(tm2[iz][j][i]+(u[iz][k][i]*wdtdr[j][k]));
				}
			}
		}
	}
	r_init((double * )r, (5*5)*5, 0.0);
	#pragma cetus parallel 
	#pragma cetus private(i, iz, j, k) 
	/*
	Disabled due to low profitability: #pragma omp parallel private(i, iz, j, k)
	*/
	{
		double (* reduce)[5][5] = (double (* )[5][5])malloc(125*sizeof (double));
		int reduce_span_0;
		int reduce_span_1;
		int reduce_span_2;
		for (reduce_span_0=0; reduce_span_0<5; reduce_span_0 ++ )
		{
			for (reduce_span_1=0; reduce_span_1<5; reduce_span_1 ++ )
			{
				
			}
			for (reduce_span_2=0; reduce_span_2<5; reduce_span_2 ++ )
			{
				reduce[reduce_span_0][reduce_span_1][reduce_span_2]=0;
			}
		}
		#pragma loop name laplacian#2 
		#pragma cetus for  
		/*
		Disabled due to low profitability: #pragma omp for
		*/
		for (k=0; k<5; k ++ )
		{
			#pragma cetus private(i, iz, j) 
			#pragma loop name laplacian#2#0 
			for (iz=0; iz<5; iz ++ )
			{
				#pragma cetus private(i, j) 
				#pragma loop name laplacian#2#0#0 
				for (j=0; j<5; j ++ )
				{
					#pragma cetus private(i) 
					#pragma loop name laplacian#2#0#0#0 
					for (i=0; i<5; i ++ )
					{
						reduce[iz][j][i]=(reduce[iz][j][i]+(u[k][j][i]*wdtdr[iz][k]));
					}
				}
			}
		}
		#pragma omp critical
		/*
		Disabled due to low profitability: #pragma omp critical
		*/
		{
			for (reduce_span_0=0; reduce_span_0<5; reduce_span_0 ++ )
			{
				for (reduce_span_1=0; reduce_span_1<5; reduce_span_1 ++ )
				{
					
				}
				for (reduce_span_2=0; reduce_span_2<5; reduce_span_2 ++ )
				{
					r[reduce_span_0][reduce_span_1][reduce_span_2]+=reduce[reduce_span_0][reduce_span_1][reduce_span_2];
				}
			}
		}
	}
	/* collocate with remaining weights and sum to complete factorization. */
	#pragma cetus private(i, j, k) 
	#pragma loop name laplacian#3 
	#pragma cetus parallel 
	/*
	Disabled due to low profitability: #pragma omp parallel for private(i, j, k)
	*/
	for (k=0; k<5; k ++ )
	{
		#pragma cetus private(i, j) 
		#pragma loop name laplacian#3#0 
		for (j=0; j<5; j ++ )
		{
			#pragma cetus private(i) 
			#pragma loop name laplacian#3#0#0 
			for (i=0; i<5; i ++ )
			{
				r[k][j][i]=((0.005*(((tm1[k][j][i]*g4m1_s[sizei][k][j][i])+(tm2[k][j][i]*g5m1_s[sizei][k][j][i]))+(r[k][j][i]*g6m1_s[sizei][k][j][i])))+((bm1_s[sizei][k][j][i]*rdtime)*u[k][j][i]));
			}
		}
	}
}
