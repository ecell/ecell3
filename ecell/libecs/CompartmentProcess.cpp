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

#include "CompartmentProcess.hpp"
#include "Vector.hpp"

LIBECS_DM_INIT(CompartmentProcess, Process); 

unsigned CompartmentProcess::getLatticeResizeCoord(unsigned aStartCoord)
{
  theComp = theSpatiocyteStepper->system2Comp(getSuperSystem());
  theVacantSpecies->resetFixedAdjoins();
  theVacantSpecies->setRadius(SubunitRadius);
  //The compartment center point (origin):
  Origin = theComp->centerPoint;
  Origin.x += OriginX*theComp->lengthX/2;
  Origin.y += OriginY*theComp->lengthY/2;
  Origin.z += OriginZ*theComp->lengthZ/2;
  setCompartmentDimension();
  for(unsigned i(0); i != theCompartmentSpecies.size(); ++i)
    {
      theCompartmentSpecies[i]->setIsOffLattice();
      theCompartmentSpecies[i]->setDimension(dimension);
      theCompartmentSpecies[i]->setVacantSpecies(theVacantSpecies);
      theCompartmentSpecies[i]->setRadius(SubunitRadius);
    }
  startCoord = aStartCoord;
  endCoord = startCoord+Filaments*Subunits;
  return endCoord-startCoord;
}

void CompartmentProcess::setCompartmentDimension()
{
  if(Length)
    {
      Subunits = (unsigned)rint(Length/(SubunitRadius*2));
    }
  if(Width)
    {
      Filaments = (unsigned)rint((Width-2*SubunitRadius)/
                                 (SubunitRadius*sqrt(3)))+1;
    }
  Width = 2*SubunitRadius+(Filaments-1)*SubunitRadius*sqrt(3); 
  Height = 2*SubunitRadius;
  if(Filaments == 1)
    {
      dimension = 1;
      Length = Subunits*SubunitRadius*2;
    }
  else
    {
      dimension = 2;
      //Add SubunitRadius for the protrusion from hexagonal arrangement:
      Length = Subunits*SubunitRadius*2+SubunitRadius;
    }
  //Normalized compartment lengths in terms of lattice voxel radius:
  nLength = Length/(VoxelRadius*2);
  nWidth = Width/(VoxelRadius*2);
  nHeight = Height/(VoxelRadius*2);

  //Actual surface area = Width*Length
}

void CompartmentProcess::initializeThird()
{
  if(!isCompartmentalized)
    {
      thePoints.resize(Filaments*Subunits);
      initializeVectors();
      initializeFilaments();
      elongateFilaments();
      connectFilaments();
      interfaceSubunits();
      isCompartmentalized = true;
    }
  theVacantSpecies->setIsPopulated();
}

void CompartmentProcess::initializeVectors()
{
  std::cout << "Length:" << nLength << std::endl;
  std::cout << "Width:" << nWidth << std::endl;
  std::cout << "Height:" << nHeight << std::endl;
  lengthStart.x = -nLength/2;
  lengthStart.y = -nWidth/2;
  lengthStart.z = -nHeight/2;
  rotate(lengthStart);
  //Translate the origin to the compartment's origin:
  lengthStart = add(lengthStart, Origin);
  lengthEnd.x = nLength/2;
  lengthEnd.y = -nWidth/2;
  lengthEnd.z = -nHeight/2;
  rotate(lengthEnd);
  //Translate the origin to the compartment's origin:
  lengthEnd = add(lengthEnd, Origin);
  //Direction vector along the compartment length:
  lengthVector = sub(lengthEnd, lengthStart);
  lengthVector = norm(lengthVector);

  widthEnd.x = nLength/2;
  widthEnd.y = nWidth/2; 
  widthEnd.z = -nHeight/2;
  rotate(widthEnd);
  widthEnd = add(widthEnd, Origin);
  //Direction vector along the compartment width:
  widthVector = sub(lengthEnd, widthEnd); 
  widthVector = norm(widthVector);

  heightEnd.x = nLength/2;
  heightEnd.y = nWidth/2;
  heightEnd.z = nHeight/2;
  rotate(heightEnd);
  heightEnd = add(heightEnd, Origin);
  //Direction vector along the compartment height:
  heightVector = sub(widthEnd, heightEnd);
  heightVector = norm(heightVector); 

  Point center(theComp->centerPoint);
  std::cout << "compCenter:" << " x:" << center.x << " y:" << center.y << " z:" << center.z << std::endl;
  std::cout << "origin:" << " x:" << Origin.x << " y:" << Origin.y << " z:" << Origin.z << std::endl;

  std::cout << "lengthStart:" << " x:" << lengthStart.x << " y:" << lengthStart.y << " z:" << lengthStart.z << std::endl;
  std::cout << "lengthEnd:" << " x:" << lengthEnd.x << " y:" << lengthEnd.y << " z:" << lengthEnd.z << std::endl;
  std::cout << "widthEnd:" << " x:" << widthEnd.x << " y:" << widthEnd.y << " z:" << widthEnd.z << std::endl;
  std::cout << "heightEnd:" << " x:" << heightEnd.x << " y:" << heightEnd.y << " z:" << heightEnd.z << std::endl;
  /*
  //The point of the first subunit:
  subunitStart = disp(lengthStart, lengthVector, nSubunitRadius);
  disp_(subunitStart, widthVector, nSubunitRadius);
  disp_(subunitStart, heightVector, nSubunitRadius);
  */
  subunitStart = lengthStart;
  
  //Set up surface vectors:
  surfaceNormal = cross(lengthVector, widthVector);
  surfaceNormal = norm(surfaceNormal);
  surfaceDisplace = dot(surfaceNormal, widthEnd);
  lengthDisplace = dot(lengthVector, lengthStart);
  lengthDisplaceOpp = dot(lengthVector, lengthEnd);
  widthDisplace = dot(widthVector, widthEnd);
  widthDisplaceOpp = dot(widthVector, lengthEnd);

}

void CompartmentProcess::rotate(Point& V)
{
  theSpatiocyteStepper->rotateX(RotateX, &V, -1);
  theSpatiocyteStepper->rotateY(RotateY, &V, -1);
  theSpatiocyteStepper->rotateZ(RotateZ, &V, -1);
}

void CompartmentProcess::initializeFilaments()
{
  addCompVoxel(0, 0, subunitStart);
  for(unsigned i(1); i != Filaments; ++i)
    {
      Point U(subunitStart);
      disp_(U, widthVector, i*nSubunitRadius*sqrt(3)); 
      if(i%2 == 1)
        {
          disp_(U, lengthVector, nSubunitRadius); 
        }
      addCompVoxel(i, 0, U);
    }
}

void CompartmentProcess::addCompVoxel(unsigned filamentIndex, 
                                   unsigned subunitIndex, Point& aPoint)
{
  unsigned aCoord(startCoord+filamentIndex*Subunits+subunitIndex);
  Voxel& aVoxel((*theLattice)[aCoord]);
  aVoxel.point = &thePoints[filamentIndex*Subunits+subunitIndex];
  *aVoxel.point = aPoint;
  aVoxel.adjoiningCoords = new unsigned[theAdjoiningCoordSize];
  aVoxel.diffuseSize = 2;
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      aVoxel.adjoiningCoords[i] = theNullCoord;
    }
  theVacantSpecies->addCompVoxel(aCoord);
}

void CompartmentProcess::elongateFilaments()
{
  for(unsigned i(0); i != Filaments; ++i)
    {
      Voxel* startVoxel(&(*theLattice)[startCoord+i*Subunits]);
      Point A(*startVoxel->point);
      for(unsigned j(1); j != Subunits; ++j)
        {
          disp_(A, lengthVector, nSubunitRadius*2);
          addCompVoxel(i, j, A);
        }
    }
}

void CompartmentProcess::connectFilaments()
{
  for(unsigned i(0); i != Subunits; ++i)
    {
      for(unsigned j(0); j != Filaments; ++j)
        { 
          if(i > 0)
            { 
              connectNorthSouth(i, j);
            }
          else if(Periodic)
            {
              connectPeriodic(j);
            }
          if(j > 0)
            {
              connectEastWest(i, j);
            }
        }
      /*
      if(Filaments > 2)
        {
          connectSeamEastWest(i);
        }
        */
      if(i > 0)
        {
          connectNwSw(i);
        }
    }
}

void CompartmentProcess::connectPeriodic(unsigned j)
{
  unsigned a(startCoord+j*Subunits+Subunits-1);
  Voxel& aVoxel((*theLattice)[a]);
  unsigned b(startCoord+j*Subunits); 
  Voxel& adjoin((*theLattice)[b]);
  aVoxel.adjoiningCoords[NORTH] = b;
  adjoin.adjoiningCoords[SOUTH] = a;
  aVoxel.adjoiningSize = 2;
  adjoin.adjoiningSize = 2;
}

void CompartmentProcess::connectNorthSouth(unsigned i, unsigned j)
{
  unsigned a(startCoord+j*Subunits+(i-1));
  Voxel& aVoxel((*theLattice)[a]);
  unsigned b(startCoord+j*Subunits+i);
  Voxel& adjoin((*theLattice)[b]);
  aVoxel.adjoiningCoords[NORTH] = b;
  adjoin.adjoiningCoords[SOUTH] = a;
  aVoxel.adjoiningSize = 2;
  adjoin.adjoiningSize = 2;
}

void CompartmentProcess::connectEastWest(unsigned i, unsigned j)
{
  unsigned a(startCoord+j*Subunits+i);
  Voxel& aVoxel((*theLattice)[a]);
  unsigned b(startCoord+(j-1)*Subunits+i); 
  Voxel& adjoin((*theLattice)[b]);
  aVoxel.adjoiningCoords[aVoxel.adjoiningSize++] = b;
  adjoin.adjoiningCoords[adjoin.adjoiningSize++] = a;
}

void CompartmentProcess::connectSeamEastWest(unsigned i)
{
  unsigned a(startCoord+i);
  Voxel& aVoxel((*theLattice)[a]);
  unsigned b(startCoord+(Filaments-1)*Subunits+i); 
  Voxel& adjoin((*theLattice)[b]);
  aVoxel.adjoiningCoords[aVoxel.adjoiningSize++] = b;
  adjoin.adjoiningCoords[adjoin.adjoiningSize++] = a;
}

void CompartmentProcess::connectNwSw(unsigned i)
{
  unsigned a(startCoord+i);
  Voxel& aVoxel((*theLattice)[a]);
  unsigned b(startCoord+(Filaments-1)*Subunits+(i-1)); 
  Voxel& adjoin((*theLattice)[b]);
  aVoxel.adjoiningCoords[aVoxel.adjoiningSize++] = b;
  adjoin.adjoiningCoords[adjoin.adjoiningSize++] = a;
}

void CompartmentProcess::interfaceSubunits()
{
  enlistInterfaceVoxels();
  enlistNonIntersectInterfaceVoxels();
  theVacantSpecies->setIsPopulated();
  theInterfaceSpecies->setIsPopulated();
}

void CompartmentProcess::enlistInterfaceVoxels()
{
  subunitInterfaces.resize(Filaments*Subunits);
  for(unsigned i(startCoord); i != endCoord; ++i)
    {
      Voxel& subunit((*theLattice)[i]);
      subunit.diffuseSize = subunit.adjoiningSize;
      Point center(*subunit.point);
      Point bottomLeft(*subunit.point);
      Point topRight(*subunit.point);
      bottomLeft.x -= nSubunitRadius+theSpatiocyteStepper->getColLength();
      bottomLeft.y -= nSubunitRadius+theSpatiocyteStepper->getLayerLength();
      bottomLeft.z -= nSubunitRadius+theSpatiocyteStepper->getRowLength();
      topRight.x += nSubunitRadius+theSpatiocyteStepper->getColLength();
      topRight.y += nSubunitRadius+theSpatiocyteStepper->getLayerLength();
      topRight.z += nSubunitRadius+theSpatiocyteStepper->getRowLength();
      unsigned blRow(0);
      unsigned blLayer(0);
      unsigned blCol(0);
      theSpatiocyteStepper->point2global(bottomLeft, blRow, blLayer, blCol);
      unsigned trRow(0);
      unsigned trLayer(0);
      unsigned trCol(0);
      theSpatiocyteStepper->point2global(topRight, trRow, trLayer, trCol);
      for(unsigned j(blRow); j <= trRow; ++j)
        {
          for(unsigned k(blLayer); k <= trLayer; ++k)
            {
              for(unsigned l(blCol); l <= trCol; ++l)
                {
                  unsigned m(theSpatiocyteStepper->global2coord(j, k, l));
                  addInterfaceVoxel(i, m);
                }
            }
        }
    }
}

void CompartmentProcess::addInterfaceVoxel(unsigned subunitCoord,
                                        unsigned voxelCoord)
{ 
  Voxel& subunit((*theLattice)[subunitCoord]);
  Point subunitPoint(*subunit.point);
  Point voxelPoint(theSpatiocyteStepper->coord2point(voxelCoord));
  double dist(getDistance(&subunitPoint, &voxelPoint));
  if(dist <= nSubunitRadius+nVoxelRadius) 
    {
      Voxel& voxel((*theLattice)[voxelCoord]);
      //theSpecies[6]->addMolecule(&voxel);
      //Insert voxel in the list of interface voxels if was not already:
      //if(voxel.id == theComp->vacantSpecies->getID())
      if(voxel.id != theInterfaceSpecies->getID())
        {
          //theSpecies[5]->addMolecule(&voxel);
          theInterfaceSpecies->addMolecule(&voxel);
        }
      subunitInterfaces[subunitCoord-startCoord].push_back(voxelCoord);
    }
}

void CompartmentProcess::enlistNonIntersectInterfaceVoxels()
{
  for(unsigned i(0); i != theInterfaceSpecies->size(); ++i)
    {
      unsigned voxelCoord(theInterfaceSpecies->getCoord(i));
      Voxel& anInterface((*theLattice)[voxelCoord]);
      for(unsigned j(0); j != theAdjoiningCoordSize; ++j)
        {
          Voxel& adjoin((*theLattice)[anInterface.adjoiningCoords[j]]);
          if(adjoin.id != theInterfaceSpecies->getID())
            {
              Point aPoint(theSpatiocyteStepper->coord2point(adjoin.coord));
              if(isInside(aPoint))
                {
                  addNonIntersectInterfaceVoxel(adjoin, aPoint);
                }
            }
        }
    }
}

bool CompartmentProcess::isInside(Point& aPoint)
{
  double dist(point2planeDist(aPoint, lengthVector, lengthDisplace));
  if(dist >= 0)
    {
      dist = point2planeDist(aPoint, lengthVector, lengthDisplaceOpp);
      if(dist <= 0)
        {
          dist = point2planeDist(aPoint, widthVector, widthDisplaceOpp);
          if(dist <=0)
            {
              dist = point2planeDist(aPoint, widthVector, widthDisplace);
              if(dist >= 0)
                {
                  return true;
                }
            }
        }
    }
  return false;
}

void CompartmentProcess::addNonIntersectInterfaceVoxel(Voxel& aVoxel,
                                                    Point& aPoint)
{
  double distA(point2planeDist(aPoint, surfaceNormal, surfaceDisplace));
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      Voxel& adjoin((*theLattice)[aVoxel.adjoiningCoords[i]]);
      if(adjoin.id != theInterfaceSpecies->getID())
        {
          Point pointB(theSpatiocyteStepper->coord2point(adjoin.coord));
          double distB(point2planeDist(pointB, surfaceNormal, surfaceDisplace));
          //if not on the same side of the plane:
          if((distA < 0) != (distB < 0))
            {
              if(abs(distA) < abs(distB))
                { 
                  //theSpecies[6]->addMolecule(&aVoxel);
                  theInterfaceSpecies->addMolecule(&aVoxel);
                  return;
                }
              else
                {
                  //theSpecies[6]->addMolecule(&adjoin);
                  theInterfaceSpecies->addMolecule(&adjoin);
                }
            }
        }
    }
}

