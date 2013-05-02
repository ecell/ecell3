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

#include <MicrotubuleProcess.hpp>

LIBECS_DM_INIT(MicrotubuleProcess, Process); 

unsigned MicrotubuleProcess::getLatticeResizeCoord(unsigned aStartCoord)
{
  const unsigned aSize(CompartmentProcess::getLatticeResizeCoord(aStartCoord));
  theMinusSpecies->resetFixedAdjoins();
  theMinusSpecies->setMoleculeRadius(DiffuseRadius);
  thePlusSpecies->resetFixedAdjoins();
  thePlusSpecies->setMoleculeRadius(DiffuseRadius);
  for(unsigned i(0); i != theKinesinSpecies.size(); ++i)
    {
      theKinesinSpecies[i]->setMoleculeRadius(DiffuseRadius);
    }
  return aSize;
}

void MicrotubuleProcess::setCompartmentDimension()
{
  if(Length)
    {
      Subunits = (unsigned)rint(Length/(2*DiffuseRadius));
    }
  Length = Subunits*2*DiffuseRadius;
  Width = Radius*2;
  Height = Radius*2;
  theDimension = 1;
  Origin.x += OriginX*theComp->lengthX/2;
  Origin.y += OriginY*theComp->lengthY/2;
  Origin.z += OriginZ*theComp->lengthZ/2;
  allocateGrid();
}


void MicrotubuleProcess::initializeThird()
{
  if(!isCompartmentalized)
    {
      thePoints.resize(endCoord-subStartCoord);
      vacStartIndex = theVacantSpecies->size();
      intStartIndex = theInterfaceSpecies->size();
      initializeVectors();
      initializeFilaments(subunitStart, Filaments, Subunits, nMonomerPitch,
                          theVacantSpecies, subStartCoord);
      elongateFilaments(theVacantSpecies, subStartCoord, Filaments, Subunits,
                        nDiffuseRadius);
      connectFilaments(subStartCoord, Filaments, Subunits);
      setGrid(theVacantSpecies, theVacGrid, subStartCoord);
      interfaceSubunits();
      isCompartmentalized = true;
    }
  theVacantSpecies->setIsPopulated();
  theInterfaceSpecies->setIsPopulated();
  theMinusSpecies->setIsPopulated();
  thePlusSpecies->setIsPopulated();
}


void MicrotubuleProcess::initializeVectors()
{ 
  //Minus end
  Minus.x = -nLength/2;
  Minus.y = 0;
  Minus.z = 0;
  //Rotated Minus end
  theSpatiocyteStepper->rotateX(RotateX, &Minus, -1);
  theSpatiocyteStepper->rotateY(RotateY, &Minus, -1);
  theSpatiocyteStepper->rotateZ(RotateZ, &Minus, -1);
  add_(Minus, Origin);
  //Direction vector from the Minus end to center
  //Direction vector from the Minus end to center
  lengthVector = sub(Origin, Minus);
  //Make direction vector a unit vector
  norm_(lengthVector);
  //Rotated Plus end
  Plus = disp(Minus, lengthVector, nLength);
  setSubunitStart();
}

void MicrotubuleProcess::setSubunitStart()
{
  Point R; //Initialize a random point on the plane attached at the minus end
  if(Minus.x != Plus.x)
    {
      R.y = 10;
      R.z = 30; 
      R.x = (Minus.x*lengthVector.x+Minus.y*lengthVector.y-R.y*lengthVector.y+
             Minus.z*lengthVector.z-R.z*lengthVector.z)/lengthVector.x;
    }
  else if(Minus.y != Plus.y)
    {
      R.x = 10; 
      R.z = 30;
      R.y = (Minus.x*lengthVector.x-R.x*lengthVector.x+Minus.y*lengthVector.y+
             Minus.z*lengthVector.z-R.z*lengthVector.z)/lengthVector.y;
    }
  else
    {
      R.x = 10; 
      R.y = 30;
      R.z = (Minus.x*lengthVector.x-R.x*lengthVector.x+Minus.y*lengthVector.y-
             R.y*lengthVector.y+Minus.z*lengthVector.z)/lengthVector.z;
    }
  //The direction vector from the minus end to the random point, R
  Point D(sub(R, Minus));
  norm_(D);
  subunitStart = disp(Minus, D, nRadius);
}

void MicrotubuleProcess::initializeFilaments(Point& aStartPoint, unsigned aRows,
                                             unsigned aCols, double aRadius,
                                             Species* aVacant,
                                             unsigned aStartCoord)
{
  Voxel* aVoxel(addCompVoxel(0, 0, aStartPoint, aVacant, aStartCoord, aCols));
  theMinusSpecies->addMolecule(aVoxel);
  Point U(aStartPoint);
  for(unsigned i(1); i < aRows; ++i)
    {
      double angle(2*M_PI/aRows);
      rotatePointAlongVector(U, Minus, lengthVector, angle);
      disp_(U, lengthVector, aRadius/(aRows-1));
      aVoxel = addCompVoxel(i, 0, U, aVacant, aStartCoord, aCols);
      theMinusSpecies->addMolecule(aVoxel);
    }
}

// y:width:rows:filaments
// z:length:cols:subunits
void MicrotubuleProcess::connectFilaments(unsigned aStartCoord,
                                          unsigned aRows, unsigned aCols)
{
  for(unsigned i(0); i != aCols; ++i)
    {
      for(unsigned j(0); j != aRows; ++j)
        { 
          if(i > 0)
            { 
              //NORTH-SOUTH
              unsigned a(aStartCoord+j*aCols+i);
              unsigned b(aStartCoord+j*aCols+(i-1));
              connectSubunit(a, b, NORTH, SOUTH);
            }
          else if(Periodic)
            {
              //periodic NORTH-SOUTH
              unsigned a(aStartCoord+j*aCols); 
              unsigned b(aStartCoord+j*aCols+aCols-1);
              connectSubunit(a, b, NORTH, SOUTH);
            }
        }
    }
}

void MicrotubuleProcess::elongateFilaments(Species* aVacant,
                                           unsigned aStartCoord,
                                           unsigned aRows,
                                           unsigned aCols,
                                           double aRadius)
{
  for(unsigned i(0); i != aRows; ++i)
    {
      Voxel* startVoxel(&(*theLattice)[aStartCoord+i*aCols]);
      Point A(*startVoxel->point);
      for(unsigned j(1); j != aCols; ++j)
        {
          disp_(A, lengthVector, aRadius*2);
          Voxel* aVoxel(addCompVoxel(i, j, A, aVacant, aStartCoord, aCols));
          if(j == aCols-1)
            {
              thePlusSpecies->addMolecule(aVoxel);
            }
        }
    }
}

//Is inside the parent compartment and confined by the length of the MT:
bool MicrotubuleProcess::isInside(Point& aPoint)
{
  double disp(point2planeDisp(aPoint, lengthVector, dot(lengthVector, Minus)));
  //Use nDifffuseRadius/2 instead of 0 because we don't want additional
  //interface voxels at the edge of the plus or minus end. So only molecules
  //hitting along the normal of the surface of the MT at the ends can bind.
  //This would avoid bias of molecule directly hitting the MT ends from the
  //sides and binding:
  if(disp > nDiffuseRadius/2)
    { 
      disp = point2planeDisp(aPoint, lengthVector, dot(lengthVector, Plus));
      if(disp < -nDiffuseRadius/2)
        {
          return true;
        }
    }
  return false;
}

bool MicrotubuleProcess::isOnAboveSurface(Point& aPoint)
{
  double disp(point2lineDisp(aPoint, lengthVector, Minus));
  if(disp >= nRadius)
    {
      return true;
    }
  return false;
}

void MicrotubuleProcess::addPlaneIntersectInterfaceVoxel(Voxel& aVoxel,
                                                       Point& aPoint)
{
  //Get the displacement from the voxel to the center line of the MT:
  double dispA(point2lineDisp(aPoint, lengthVector, Minus));
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      Voxel& adjoin((*theLattice)[aVoxel.adjoiningCoords[i]]);
      //if(getID(adjoin) != theInterfaceSpecies->getID())
      if(theSpecies[getID(adjoin)]->getIsCompVacant())
        {
          Point pointB(theSpatiocyteStepper->coord2point(adjoin.coord));
          //Get the displacement from the adjoin to the center line of the MT:
          double dispB(point2lineDisp(pointB, lengthVector, Minus));
          //If not on the same side of the MT surface, or one of it is on
          //the MT surface while the other is not:
          if((dispA < nRadius) != (dispB < nRadius))
            {
              //If the voxel is nearer to the MT surface:
              if(abs(dispA-nRadius) < abs(dispB-nRadius))
                {
                  addInterfaceVoxel(aVoxel, aPoint);
                  return;
                }
              //If the adjoin is nearer to the MT surface:
              else
                {
                  addInterfaceVoxel(adjoin, pointB);
                }
            }
        }
    }
}

