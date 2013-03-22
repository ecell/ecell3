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


#ifndef __Vector_hpp
#define __Vector_hpp

#include "SpatiocyteCommon.hpp"

Point cross(Point& L, Point& R)
{
  Point V;
  V.x  = L.y * R.z;
  V.y  = L.z * R.x;
  V.z  = L.x * R.y;
  V.x -= R.y * L.z;
  V.y -= R.z * L.x;
  V.z -= R.x * L.y;
  return V;
}

Point sub(Point& L, Point& R)
{
  Point V;
  V.x = L.x - R.x;
  V.y = L.y - R.y;
  V.z = L.z - R.z;
  return V;
}

void sub_(Point& L, Point& R)
{
  L.x -= R.x;
  L.y -= R.y;
  L.z -= R.z;
}

Point add(Point& L, Point& R)
{
  Point V;
  V.x = L.x + R.x;
  V.y = L.y + R.y;
  V.z = L.z + R.z;
  return V;
}

void add_(Point& L, Point& R)
{
  L.x += R.x;
  L.y += R.y;
  L.z += R.z;
}

Point norm(Point& P)
{
  double denom(sqrt(P.x*P.x+P.y*P.y+P.z*P.z));
  Point V;
  V.x = P.x/denom;
  V.y = P.y/denom;
  V.z = P.z/denom;
  return V;
}

Point disp(Point& P, Point& V, double dist)
{
  Point A;
  A.x = P.x + V.x*dist;
  A.y = P.y + V.y*dist;
  A.z = P.z + V.z*dist;
  return A;
}

void disp_(Point& P, Point& V, double dist)
{
  P.x += V.x*dist;
  P.y += V.y*dist;
  P.z += V.z*dist;
}

double dot(Point& L, Point& R)
{
  return L.x*R.x + L.y*R.y + L.z*R.z;
}

//Get the shortest distance from a point, P to a plane given by normal N and
//displacement, m:
double point2planeDist(Point& P, Point& N, double m)
{
  return dot(P, N) - m;
}

double abs(double a)
{
  return sqrt(a*a);
}

/*
//Get the intersection point for the line connecting A to B with a plane having
//the normal N and the distance from origin m:
Point intersectLinePlane(Point& A, Point& B, Point& N, double m)
{
  Point V;
*/


#endif /* __Vector_hpp */
