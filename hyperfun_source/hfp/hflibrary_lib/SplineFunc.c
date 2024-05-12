/*==============================================================================

 Copyright 1998, 1999 Valery Adzhiev, Alexander Pasko, Ken Yoshikawa
 Copyright 2003-2004 Benjamin Schmitt

 This Work or file is part of the greater total Work, software or group of
 files named HyperFun Polygonizer.

 HyperFun Polygonizer can be redistributed and/or modified under the terms
 of the CGPL, The Common Good Public License as published by and at CGPL.org
 (http://CGPL.org).  It is released under version 1.0 Beta of the License
 until the 1.0 version is released after which either version 1.0 of the
 License, or (at your option) any later version can be applied.

 THIS WORK, OR SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
 WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED (See the
 CGPL, The Common Good Public License for more information.)

 You should have received a copy of the CGPL along with HyperFun Polygonizer;
 if not, see -  http://CGPL.org to get a copy of the License.

==============================================================================*/


#include <math.h>
#include "SplineFunc.h"
#include "hflib.h"

static double Combin[21][21] = {{1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000,-1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 2.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 3.000000, 3.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 4.000000, 6.000000, 4.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 5.000000, 10.000000, 10.000000, 5.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 6.000000, 15.000000, 20.000000, 15.000000, 6.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 7.000000, 21.000000, 35.000000, 35.000000, 21.000000, 7.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 8.000000, 28.000000, 56.000000, 70.000000, 56.000000, 28.000000, 8.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 9.000000, 36.000000, 84.000000, 126.000000, 126.000000, 84.000000, 36.000000, 9.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 10.000000, 45.000000, 120.000000, 210.000000, 252.000000, 210.000000, 120.000000, 45.000000, 10.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 11.000000, 55.000000, 165.000000, 330.000000, 462.000000, 462.000000, 330.000000, 165.000000, 55.000000, 11.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 12.000000, 66.000000, 220.000000, 495.000000, 792.000000, 924.000000, 792.000000, 495.000000, 220.000000, 66.000000, 12.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 13.000000, 78.000000, 286.000000, 715.000000, 1287.000000, 1716.000000, 1716.000000, 1287.000000, 715.000000, 286.000000, 78.000000, 13.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 14.000000, 91.000000, 364.000000, 1001.000000, 2002.000000, 3003.000000, 3432.000000, 3003.000000, 2002.000000, 1001.000000, 364.000000, 91.000000, 14.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 15.000000, 105.000000, 455.000000, 1365.000000, 3003.000000, 5005.000000, 6435.000000, 6435.000000, 5005.000000, 3003.000000, 1365.000000, 455.000000, 105.000000, 15.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 16.000000, 120.000000, 560.000000, 1820.000000, 4368.000000, 8008.000000, 11440.000000, 12870.000000, 11440.000000, 8008.000000, 4368.000000, 1820.000000, 560.000000, 120.000000, 16.000000, 1.000000, -1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 17.000000, 136.000000, 680.000000, 2380.000000, 6188.000000, 12376.000000, 19448.000000, 24310.000000, 24310.000000, 19448.000000, 12376.000000, 6188.000000, 2380.000000, 680.000000, 136.000000, 17.000000, 1.000000, -1.000000, -1.000000, -1.000000},
{1.000000, 18.000000, 153.000000, 816.000000, 3060.000000, 8568.000000, 18564.000000, 31824.000000, 43758.000000, 48620.000000, 43758.000000, 31824.000000, 18564.000000, 8568.000000, 3060.000000, 816.000000, 153.000000, 18.000000, 1.000000, -1.000000, -1.000000},
{1.000000, 19.000000, 171.000000, 969.000000, 3876.000000, 11628.000000, 27132.000000, 50388.000000, 75582.000000, 92378.000000, 92378.000000, 75582.000000, 50388.000000, 27132.000000, 11628.000000, 3876.000000, 969.000000, 171.000000, 19.000000, 1.000000, -1.000000},
{1.000000, 20.000000, 190.000000, 1140.000000, 4845.000000, 15504.000000, 38760.000000, 77520.000000, 125970.000000, 167960.000000, 184756.000000, 167960.000000, 125970.000000, 77520.000000, 38760.000000, 15504.000000, 4845.000000, 1140.000000, 190.000000, 20.000000, 1.000000}};


/*******************************************************************************/

double Bernstein(int i,int n, double t)
{
  return Combin[n][i]*pow((1.-t),n-i)*pow(t,i);
}


/*******************************************************************************/

double Knot_u(int i)
{
  if(i<KnotOrder_u) return 0.0;
  else if(i>KnotNbSom_u) return (double)(KnotNbSom_u-KnotOrder_u+2);
  else return (double)(i-KnotOrder_u +1);
}


/*******************************************************************************/

double RecursSpline_u(int i,int n,double t){
  double B,T;
  if(n==1) {
    if( ((Knot_u(i)<=t) && (t<Knot_u(i+1)))
	|| ((t==Knot_u(i+1)) &&  (i==KnotNbSom_u))) B=1.0;
    else B=0.0;
  }
  else {
    B=0.0;
    T=Knot_u(i+n-1)-Knot_u(i);
    if(T!=0.0) B=(t-Knot_u(i))*RecursSpline_u(i,n-1,t)/T;

    T=Knot_u(i+n)-Knot_u(i+1);
    if(T!=0.0) B=B+(Knot_u(i+n)-t)*RecursSpline_u(i+1,n-1,t)/T;
    /* return B; */
  }
  return B;
}


/*******************************************************************************/

double Knot_v(int i)
{
  if(i<KnotOrder_v) return 0.0;
  else if(i>KnotNbSom_v) return (double)(KnotNbSom_v-KnotOrder_v+2);
  else return (double)(i-KnotOrder_v +1);
}


/*******************************************************************************/

double RecursSpline_v(int i,int n,double t){
  double B,T;
  if(n==1) {
    if( ((Knot_v(i)<=t) && (t<Knot_v(i+1)))
	|| ((t==Knot_v(i+1)) &&  (i==KnotNbSom_v))) B=1.0;
    else B=0.0;
  }
  else {
    B=0.0;
    T=Knot_v(i+n-1)-Knot_v(i);
    if(T!=0.0) B=(t-Knot_v(i))*RecursSpline_v(i,n-1,t)/T;

    T=Knot_v(i+n)-Knot_v(i+1);
    if(T!=0.0) B=B+(Knot_v(i+n)-t)*RecursSpline_v(i+1,n-1,t)/T;
    /* return B; */
  }
  return B;
}


/*******************************************************************************/

double Knot_w(int i)
{
  if(i<KnotOrder_w) return 0.0;
  else if(i>KnotNbSom_w) return (double)(KnotNbSom_w-KnotOrder_w+2);
  else return (double)(i-KnotOrder_w +1);
}


/*******************************************************************************/

double RecursSpline_w(int i,int n,double t){
  double B,T;
  if(n==1) {
    if( ((Knot_w(i)<=t) && (t<Knot_w(i+1)))
	|| ((t==Knot_w(i+1)) &&  (i==KnotNbSom_w))) B=1.0;
    else B=0.0;
  }
  else {
    B=0.0;
    T=Knot_w(i+n-1)-Knot_w(i);
    if(T!=0.0) B=(t-Knot_w(i))*RecursSpline_w(i,n-1,t)/T;

    T=Knot_w(i+n)-Knot_w(i+1);
    if(T!=0.0) B=B+(Knot_w(i+n)-t)*RecursSpline_w(i+1,n-1,t)/T;
    /* return B; */

  }
  return B;

}
