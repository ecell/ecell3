//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2006-2009 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Satya Arjunan <satya.arjunan@gmail.com>
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//


#ifndef __SpatiocytePolymer_hpp
#define __SpatiocytePolymer_hpp

#include "SpatiocyteCommon.hpp"

void reverseDcm( double *dcm1, double *dcm2 )
{
  dcm2[0] = -dcm1[0];
  dcm2[1] = -dcm1[1];
  dcm2[2] = -dcm1[2];
  dcm2[3] = -dcm1[3];
  dcm2[4] = -dcm1[4];
  dcm2[5] = -dcm1[5];
  dcm2[6] = dcm1[6];
  dcm2[7] = dcm1[7];
  dcm2[8] = dcm1[8];
}

void reverseYpr( double *dcm1, double *dcm2 )
{
  dcm2[0] = dcm1[0];
  dcm2[1] = -dcm1[1];
  dcm2[2] = dcm1[2];
  dcm2[3] = -dcm1[3];
  dcm2[4] = dcm1[4];
  dcm2[5] = -dcm1[5];
  dcm2[6] = dcm1[6];
  dcm2[7] = -dcm1[7];
  dcm2[8] = dcm1[8];
}

bool compareDcm( double* dcm1, double* dcm2 )
{
  for( int i(0); i!=9; ++i )
    {
      if( dcm1[i] != dcm2[i] )
        {
          return false;
        }
    }
  return true;
}

void ypr2yprR(double *inYpr, double *reversedYpr)
{
  double cf,cq,cy,sf,sq,sy; 
  cf=cos(inYpr[0]);
  cq=cos(inYpr[1]);
  cy=cos(inYpr[2]);
  sf=sin(inYpr[0]);
  sq=sin(inYpr[1]);
  sy=sin(inYpr[2]);
  reversedYpr[0] = atan2(sy*sq*cf-cy*sf,cq*cf);
  reversedYpr[1] = -asin(-cy*sq*cf-sy*sf);
  reversedYpr[2] = atan2(-cy*sq*sf+sy*cf,cq*cy);
}

void rotY(double *dcm,double theta)
{
  dcm[0]=cos(theta);
  dcm[1]=0;
  dcm[2]=-sin(theta);
  dcm[3]=0;
  dcm[4]=1;
  dcm[5]=0;
  dcm[6]=sin(theta);
  dcm[7]=0;
  dcm[8]=cos(theta);
}

void rotate( double *dcm, double *x, double *y, double *z )
{
  double x0(*x);
  double y0(*y);
  double z0(*z);
  *x = dcm[0]*x0 + dcm[1]*y0 + dcm[2]*z0;
  *y = dcm[3]*x0 + dcm[4]*y0 + dcm[5]*z0;
  *z = dcm[6]*x0 + dcm[7]*y0 + dcm[8]*z0;
}

void getOneDcm(double *Dcm)
{
  Dcm[0]=Dcm[4]=Dcm[8]=1;
  Dcm[1]=Dcm[2]=Dcm[3]=Dcm[5]=Dcm[6]=Dcm[7]=0;
}

void ypr2dcm(double *ypr, double *dcm)
{
  double sy(sin(ypr[0]));
  double sp(sin(ypr[1]));
  double sr(sin(ypr[2]));
  double cy(cos(ypr[0]));
  double cp(cos(ypr[1]));
  double cr(cos(ypr[2]));
  dcm[0]=cp*cy;
  dcm[1]=cp*sy;
  dcm[2]=-sp;
  dcm[3]=sr*sp*cy-cr*sy;
  dcm[4]=sr*sp*sy+cr*cy;
  dcm[5]=cp*sr;
  dcm[6]=cr*sp*cy+sr*sy;
  dcm[7]=cr*sp*sy-sr*cy;
  dcm[8]=cp*cr;
}

void dcm2ypr(double *dcm, double *ypr)
{
  ypr[0] = atan2(dcm[1], dcm[0]);
  ypr[1] = asin(-dcm[2]);
  ypr[2] = atan2(dcm[5], dcm[8]);
}

void dcmXdcm(double *dcm1, double *dcm2, double *dcm3)
{
  double work[9];
  work[0]=dcm1[0]*dcm2[0]+dcm1[1]*dcm2[3]+dcm1[2]*dcm2[6];
  work[1]=dcm1[0]*dcm2[1]+dcm1[1]*dcm2[4]+dcm1[2]*dcm2[7];
  work[2]=dcm1[0]*dcm2[2]+dcm1[1]*dcm2[5]+dcm1[2]*dcm2[8];
  work[3]=dcm1[3]*dcm2[0]+dcm1[4]*dcm2[3]+dcm1[5]*dcm2[6];
  work[4]=dcm1[3]*dcm2[1]+dcm1[4]*dcm2[4]+dcm1[5]*dcm2[7];
  work[5]=dcm1[3]*dcm2[2]+dcm1[4]*dcm2[5]+dcm1[5]*dcm2[8];
  work[6]=dcm1[6]*dcm2[0]+dcm1[7]*dcm2[3]+dcm1[8]*dcm2[6];
  work[7]=dcm1[6]*dcm2[1]+dcm1[7]*dcm2[4]+dcm1[8]*dcm2[7];
  work[8]=dcm1[6]*dcm2[2]+dcm1[7]*dcm2[5]+dcm1[8]*dcm2[8];
  dcm3[0]=work[0];
  dcm3[1]=work[1];
  dcm3[2]=work[2];
  dcm3[3]=work[3];
  dcm3[4]=work[4];
  dcm3[5]=work[5];
  dcm3[6]=work[6];
  dcm3[7]=work[7];
  dcm3[8]=work[8];
}

void dcmXdcmt(double *Dcm1,double *Dcmt,double *dcm3)
{
  dcm3[0]=Dcm1[0]*Dcmt[0]+Dcm1[1]*Dcmt[1]+Dcm1[2]*Dcmt[2];
  dcm3[1]=Dcm1[0]*Dcmt[3]+Dcm1[1]*Dcmt[4]+Dcm1[2]*Dcmt[5];
  dcm3[2]=Dcm1[0]*Dcmt[6]+Dcm1[1]*Dcmt[7]+Dcm1[2]*Dcmt[8];
  dcm3[3]=Dcm1[3]*Dcmt[0]+Dcm1[4]*Dcmt[1]+Dcm1[5]*Dcmt[2];
  dcm3[4]=Dcm1[3]*Dcmt[3]+Dcm1[4]*Dcmt[4]+Dcm1[5]*Dcmt[5];
  dcm3[5]=Dcm1[3]*Dcmt[6]+Dcm1[4]*Dcmt[7]+Dcm1[5]*Dcmt[8];
  dcm3[6]=Dcm1[6]*Dcmt[0]+Dcm1[7]*Dcmt[1]+Dcm1[8]*Dcmt[2];
  dcm3[7]=Dcm1[6]*Dcmt[3]+Dcm1[7]*Dcmt[4]+Dcm1[8]*Dcmt[5];
  dcm3[8]=Dcm1[6]*Dcmt[6]+Dcm1[7]*Dcmt[7]+Dcm1[8]*Dcmt[8];
}

void rotXrotY(double *dcm, double x, double y)
{
  double tmp[9];
  double cx(cos(x));
  double sx(sin(x)); 
  double cy(cos(y));
  double sy(sin(y));
  tmp[0]=dcm[0]*cy-dcm[6]*sy;
  tmp[1]=dcm[1]*cy-dcm[7]*sy;
  tmp[2]=dcm[2]*cy-dcm[8]*sy;
  tmp[3]=dcm[3]*cx+dcm[6]*cy*sx+dcm[0]*sx*sy;
  tmp[4]=dcm[4]*cx+dcm[7]*cy*sx+dcm[1]*sx*sy;
  tmp[5]=dcm[5]*cx+dcm[8]*cy*sx+dcm[2]*sx*sy;
  tmp[6]=dcm[6]*cx*cy-dcm[3]*sx+dcm[0]*cx*sy;
  tmp[7]=dcm[7]*cx*cy-dcm[4]*sx+dcm[1]*cx*sy;
  tmp[8]=dcm[8]*cx*cy-dcm[5]*sx+dcm[2]*cx*sy;
  dcm[0]=tmp[0];
  dcm[1]=tmp[1];
  dcm[2]=tmp[2];
  dcm[3]=tmp[3];
  dcm[4]=tmp[4];
  dcm[5]=tmp[5];
  dcm[6]=tmp[6];
  dcm[7]=tmp[7];
  dcm[8]=tmp[8];
}

double pinPoint(double *pt1, double *pt2, char surface, double radius,
                double hlength, double *dcm)
{
  double scale,dist,pt2new[3],diff;
  int ecloc; 
  if(pt2[0]<-hlength) ecloc=1;
  else if(pt2[0]<=hlength) ecloc=2;
  else ecloc=3;
  if(ecloc==2)
    {
      scale=radius/sqrt(pt2[1]*pt2[1]+pt2[2]*pt2[2]);
      pt2new[0]=pt2[0];
    }
  else if(ecloc==1)
    {
      scale=radius/sqrt((pt2[0]+hlength)*(pt2[0]+hlength)+
                        pt2[1]*pt2[1]+pt2[2]*pt2[2]);
      pt2new[0]=(pt2[0]+hlength)*scale-hlength;
    }
  else if(ecloc==3)
    {
      scale=radius/sqrt((pt2[0]-hlength)*(pt2[0]-hlength)+
                        pt2[1]*pt2[1]+pt2[2]*pt2[2]);
      pt2new[0]=(pt2[0]-hlength)*scale+hlength;
    }
  if(!(scale>0)) return -1;
  pt2new[1]=scale*pt2[1]; // multiply pt2 by scaling factor
  pt2new[2]=scale*pt2[2];
  diff=sqrt((pt2new[0]-pt2[0])*(pt2new[0]-pt2[0])+
            (pt2new[1]-pt2[1])*(pt2new[1]-pt2[1])+
            (pt2new[2]-pt2[2])*(pt2new[2]-pt2[2]));
  pt2[0]=pt2new[0];
  pt2[1]=pt2new[1];
  pt2[2]=pt2new[2]; 
  dcm[0]=pt2[0]-pt1[0]; // local x vector points in vector direction
  dcm[1]=pt2[1]-pt1[1];
  dcm[2]=pt2[2]-pt1[2];
  dist=sqrt(dcm[0]*dcm[0]+dcm[1]*dcm[1]+dcm[2]*dcm[2]);
  if(!(dist>0)) return -1;
  dcm[0]/=dist;
  dcm[1]/=dist;
  dcm[2]/=dist; 
  if(ecloc==1&&pt1[0]<=-hlength) dcm[6]=-0.5*(pt1[0]+pt2[0])-hlength;
  else if(ecloc==3&&pt1[0]>=hlength) dcm[6]=-0.5*(pt1[0]+pt2[0])+hlength;
  else if(ecloc==2&&pt1[0]>=-hlength&&pt1[0]<=hlength) dcm[6]=0;
  else dcm[6]=0.5*((pt1[1]+pt2[1])* (pt2[1]-pt1[1])+
                   (pt1[2]+pt2[2])*(pt2[2]-pt1[2]))/(pt2[0]-pt1[0]);
  dcm[7]=-0.5*(pt1[1]+pt2[1]);
  dcm[8]=-0.5*(pt1[2]+pt2[2]);
  dist=dcm[0]*dcm[6]+dcm[1]*dcm[7]+dcm[2]*dcm[8];
  dcm[6]-=dist*dcm[0];
  dcm[7]-=dist*dcm[1];
  dcm[8]-=dist*dcm[2];
  dist=sqrt(dcm[6]*dcm[6]+dcm[7]*dcm[7]+dcm[8]*dcm[8]);	// length vector
  if(!(dist>0)) return -1;
  dcm[6]/=dist;
  dcm[7]/=dist;
  dcm[8]/=dist; 
  dcm[3]=dcm[7]*dcm[2]-dcm[8]*dcm[1]; // local y is z cross x
  dcm[4]=dcm[8]*dcm[0]-dcm[6]*dcm[2];
  dcm[5]=dcm[6]*dcm[1]-dcm[7]*dcm[0];
  dist=sqrt(dcm[3]*dcm[3]+dcm[4]*dcm[4]+dcm[5]*dcm[5]);
  if(!(dist>0)) return -1;
  dcm[3]/=dist;
  dcm[4]/=dist;
  dcm[5]/=dist;
  return diff;
} 

#endif /* __SpatiocytePolymer_hpp */
