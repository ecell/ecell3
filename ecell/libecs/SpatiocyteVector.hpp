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


#ifndef __SpatiocyteVector_hpp
#define __SpatiocyteVector_hpp

#include <SpatiocyteCommon.hpp>

Point cross(const Point& L, const Point& R)
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

Point sub(const Point& L, const Point& R)
{
  Point V;
  V.x = L.x - R.x;
  V.y = L.y - R.y;
  V.z = L.z - R.z;
  return V;
}

void sub_(Point& L, const Point& R)
{
  L.x -= R.x;
  L.y -= R.y;
  L.z -= R.z;
}

Point add(const Point& L, const Point& R)
{
  Point V;
  V.x = L.x + R.x;
  V.y = L.y + R.y;
  V.z = L.z + R.z;
  return V;
}

void add_(Point& L, const Point& R)
{
  L.x += R.x;
  L.y += R.y;
  L.z += R.z;
}

Point norm(const Point& P)
{
  double denom(sqrt(P.x*P.x+P.y*P.y+P.z*P.z));
  Point V;
  V.x = P.x/denom;
  V.y = P.y/denom;
  V.z = P.z/denom;
  return V;
}

void norm_(Point& P)
{
  double denom(sqrt(P.x*P.x+P.y*P.y+P.z*P.z));
  P.x /= denom;
  P.y /= denom;
  P.z /= denom;
}

Point disp(const Point& P, const Point& V, const double dist)
{
  Point A;
  A.x = P.x + V.x*dist;
  A.y = P.y + V.y*dist;
  A.z = P.z + V.z*dist;
  return A;
}

void disp_(Point& P, const Point& V, const double dist)
{
  P.x += V.x*dist;
  P.y += V.y*dist;
  P.z += V.z*dist;
}

Point mult(const Point& P, const double dist)
{
  Point A;
  A.x = P.x*dist;
  A.y = P.y*dist;
  A.z = P.z*dist;
  return A;
}

double dot(const Point& L, const Point& R)
{
  return L.x*R.x + L.y*R.y + L.z*R.z;
}

double distance(const Point& L, const Point& R)
{
  return sqrt((L.x-R.x)*(L.x-R.x) + (L.y-R.y)*(L.y-R.y) + (L.z-R.z)*(L.z-R.z));
}

//Get the shortest distance from a point, P to a plane given by normal N and
//displacement, m:
double point2planeDisp(const Point& P, const Point& N, const double m)
{
  return dot(P, N) - m;
}

//Get the shortest distance from a point, P to a line defined by the direction
//vector, N that passes through a point, Q:
double point2lineDisp(const Point& P, const Point& N, const Point& Q)
{
  double t((dot(P, N) - dot(Q, N))/dot(N, N));
  Point A(mult(N, t));
  add_(A, Q);
  return distance(P, A);
}

double abs(const double a)
{
  return sqrt(a*a);
}

//Return the result when the point P(x,y,z) is rotated about the line through
//C(a,b,c) with unit direction vector N⟨u,v,w⟩ by the angle θ.
void rotatePointAlongVector(Point& P, const Point& C, const Point& N,
                            const double angle)
{
  double x(P.x);
  double y(P.y);
  double z(P.z);
  double a(C.x);
  double b(C.y);
  double c(C.z);
  double u(N.x);
  double v(N.y);
  double w(N.z);
  double u2(u*u);
  double v2(v*v);
  double w2(w*w);
  double cosT(cos(angle));
  double oneMinusCosT(1-cosT);
  double sinT(sin(angle));
  double xx((a*(v2 + w2) - u*(b*v + c*w - u*x - v*y - w*z)) * oneMinusCosT
                + x*cosT + (-c*v + b*w - w*y + v*z)*sinT);
  double yy((b*(u2 + w2) - v*(a*u + c*w - u*x - v*y - w*z)) * oneMinusCosT
                + y*cosT + (c*u - a*w + w*x - u*z)*sinT);
  double zz((c*(u2 + v2) - w*(a*u + b*v - u*x - v*y - w*z)) * oneMinusCosT
                + z*cosT + (-b*u + a*v - v*x + u*y)*sinT);
  P.x = xx;
  P.y = yy;
  P.z = zz;
}

#endif /* __SpatiocyteVector_hpp */
