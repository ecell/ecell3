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

#include <ErythrocyteProcess.hpp>

LIBECS_DM_INIT(ErythrocyteProcess, Process); 

void ErythrocyteProcess::initializeThird()
{
  if(!isCompartmentalized)
    {
      theComp = theSpatiocyteStepper->system2Comp(getSuperSystem());
      C = theComp->centerPoint;
      initializeDirectionVectors();
      initializeProtofilaments();
      theSpectrinSpecies->setIsPopulated();
      theVertexSpecies->setIsPopulated();
      isCompartmentalized = true;
    }
}

void ErythrocyteProcess::populateMolecules()
{
  for(unsigned int i(0); i != filamentCoords.size(); ++i)
    {
      theSpectrinSpecies->addMolecule(&(*theLattice)[filamentCoords[i]]);
    }
  theSpectrinSpecies->setIsPopulated();
  cout << "size of spectrin:" << theSpectrinSpecies->size() << std::endl;
}

void ErythrocyteProcess::initializeDirectionVectors()
{ 
  //Direction vector along rotated positive y-axis
  Point A;
  A.x = 0; 
  A.y = -theComp->lengthY/2;
  A.z = 0;
  theSpatiocyteStepper->rotateX(theComp->rotateX, &A, -1);
  theSpatiocyteStepper->rotateY(theComp->rotateY, &A, -1);
  theSpatiocyteStepper->rotateZ(theComp->rotateZ, &A, -1);
  A.x += C.x;
  A.y += C.y;
  A.z += C.z;
  Y.x = C.x-A.x;
  Y.y = C.y-A.y;
  Y.z = C.z-A.z;
  normalize(Y);

  //Direction vector along rotated positive z-axis
  A.x = 0; 
  A.y = 0;
  A.z = -theComp->lengthZ/2;
  theSpatiocyteStepper->rotateX(theComp->rotateX, &A, -1);
  theSpatiocyteStepper->rotateY(theComp->rotateY, &A, -1);
  theSpatiocyteStepper->rotateZ(theComp->rotateZ, &A, -1);
  A.x += C.x;
  A.y += C.y;
  A.z += C.z;
  Z.x = C.x-A.x;
  Z.y = C.y-A.y;
  Z.z = C.z-A.z;
  normalize(Z);

  //Direction vector along rotated north-east
  A.x = C.x+5*Y.x;
  A.y = C.y+5*Y.y;
  A.z = C.z+5*Y.z;
  Point B(A);
  rotatePointAlongVector(Z, C, A, M_PI/3); 
  R.x = A.x-C.x;
  R.y = A.y-C.y;
  R.z = A.z-C.z;
  normalize(R);

  //Direction vector along rotated north-west
  rotatePointAlongVector(Z, C, B, -M_PI/3); 
  L.x = B.x-C.x;
  L.y = B.y-C.y;
  L.z = B.z-C.z;
  normalize(L);
}

void ErythrocyteProcess::normalize(Point& P)
{
  double Norm(sqrt(P.x*P.x+P.y*P.y+P.z*P.z));
  P.x /= Norm;
  P.y /= Norm;
  P.z /= Norm;
}

void ErythrocyteProcess::initializeProtofilaments()
{
  Species* aVacant(theComp->vacantSpecies);
  for(unsigned int i(0); i != aVacant->size(); ++i)
    {
      unsigned int aCoord(aVacant->getCoord(i));
      Point aPoint(theSpatiocyteStepper->coord2point(aCoord));
      unsigned int cnt(getIntersectCount(aPoint, aCoord));
      if(cnt == 1)
        {
          theSpectrinSpecies->addMolecule(&(*theLattice)[aCoord]);
        }
      else if(cnt > 1)
        {
          theVertexSpecies->addMolecule(&(*theLattice)[aCoord]);
        }
    }
}

unsigned int ErythrocyteProcess::getIntersectCount(Point& aPoint,
                                                   unsigned int aCoord)
{ 
  unsigned int cnt(0);
  if(isOnUpperPlanes(aPoint, aCoord, Y))
    {
      ++cnt;
    } 
  else if(isOnLowerPlanes(aPoint, aCoord, Y))
    {
      ++cnt;
    }
  if(isOnUpperPlanes(aPoint, aCoord, R))
    {
      ++cnt;
    } 
  else if(isOnLowerPlanes(aPoint, aCoord, R))
    {
      ++cnt;
    }
  if(isOnUpperPlanes(aPoint, aCoord, L))
    {
      ++cnt;
    } 
  else if(isOnLowerPlanes(aPoint, aCoord, L))
    {
      ++cnt;
    }
  return cnt;
}

bool ErythrocyteProcess::isOnUpperPlanes(Point& aPoint, unsigned int aCoord,
                                         Point& N)
{ 
  double currLength(0);
  while(currLength < theComp->lengthY/2)
    {
      Point A;
      A.x = C.x+currLength*N.x;
      A.y = C.y+currLength*N.y;
      A.z = C.z+currLength*N.z;
      if(isOnPlane(N, A, aPoint, aCoord)) 
        {
          return true;
        }
      currLength += TriangleAltitude;
    }
  return false;
}

bool ErythrocyteProcess::isOnLowerPlanes(Point& aPoint, unsigned int aCoord,
                                         Point& N)
{ 
  double currLength(-TriangleAltitude);
  while(currLength > -theComp->lengthY/2)
    {
      Point A;
      A.x = C.x+currLength*N.x;
      A.y = C.y+currLength*N.y;
      A.z = C.z+currLength*N.z;
      if(isOnPlane(N, A, aPoint, aCoord)) 
        {
          return true;
        }
      currLength -= TriangleAltitude;
    }
  return false;
}

//N = unit normal vector of the plane
//P = a point on the plane
//T = a target point
bool ErythrocyteProcess::isOnPlane(Point& N, Point& P, Point& T,
                                   unsigned int aCoord)
{
  if(isInsidePlane(N, P, T))
    {
      Voxel& aVoxel((*theLattice)[aCoord]);
      for(unsigned int j(0); j != aVoxel.adjoiningSize; ++j)
        {
          Point adPoint(theSpatiocyteStepper->coord2point(
                            aVoxel.adjoiningCoords[j]));
          if(!isInsidePlane(N, P, adPoint))
            {
              return true;
            }
        }
    }
  return false;
}

//N = unit normal vector of the plane
//P = a point on the plane
//T = a target point
bool ErythrocyteProcess::isInsidePlane(Point& N, Point& P, Point& T)
{
  double dist((P.x-T.x)*N.x+(P.y-T.y)*N.y+(P.z-T.z)*N.z);
  if(dist < 0)
    {
      return true;
    }
  return false;
}

/*
 * The function returns the result when the point T is rotated about the line
 * through P with unit direction vector D by the angle θ.
 * */
void ErythrocyteProcess::rotatePointAlongVector(Point& D, Point& P, Point& T,
                                                double angle)
{
  double u(D.x*D.x);
  double v(D.y*D.y);
  double w(D.z*D.z);
  double cosT(cos(angle));
  double oneMinusCosT(1-cosT);
  double sinT(sin(angle));
  double xx((P.x*(v + w) - D.x*
             (P.y*D.y + P.z*D.z - D.x*T.x - D.y*T.y - D.z*T.z))*oneMinusCosT +
            T.x*cosT + (-P.z*D.y + P.y*D.z - D.z*T.y + D.y*T.z)*sinT);
  double yy((P.y*(u + w) - D.y*
             (P.x*D.x + P.z*D.z - D.x*T.x - D.y*T.y - D.z*T.z))*oneMinusCosT +
            T.y*cosT + (P.z*D.x - P.x*D.z + D.z*T.x - D.x*T.z)*sinT);
  double zz((P.z*(u + v) - D.z*(P.x*D.x + P.y*D.y - D.x*T.x - D.y*T.y -
                                D.z*T.z)) * oneMinusCosT + 
            T.z*cosT + (-P.y*D.x + P.x*D.y - D.y*T.x + D.x*T.y)*sinT);
  T.x = xx;
  T.y = yy;
  T.z = zz;
}

