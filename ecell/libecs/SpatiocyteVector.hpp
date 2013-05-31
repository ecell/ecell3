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

#include <libecs/SpatiocyteCommon.hpp>

namespace libecs
{

Point cross(const Point& L, const Point& R);
Point sub(const Point& L, const Point& R);
void sub_(Point& L, const Point& R);
Point add(const Point& L, const Point& R);
void add_(Point& L, const Point& R);
Point norm(const Point& P);
void norm_(Point& P);
Point disp(const Point& P, const Point& V, const double dist);
void disp_(Point& P, const Point& V, const double dist);
Point mult(const Point& P, const double dist);
double dot(const Point& L, const Point& R);
double distance(const Point& L, const Point& R);

//Get the shortest distance from a point, P to a plane given by normal N and
//displacement, m:
double point2planeDisp(const Point& P, const Point& N, const double m);

//Get the shortest distance from a point, P to a line defined by the direction
//vector, N that passes through a point, Q:
double point2lineDisp(const Point& P, const Point& N, const Point& Q);
double abs(const double a);

//Return the result when the point P(x,y,z) is rotated about the line through
//C(a,b,c) with unit direction vector N⟨u,v,w⟩ by the angle θ.
void rotatePointAlongVector(Point& P, const Point& C, const Point& N,
                            const double angle);

}

#endif /* __SpatiocyteVector_hpp */
