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

#include <CompartmentProcess.hpp>
#include <SpatiocyteVector.hpp>

LIBECS_DM_INIT(CompartmentProcess, Process); 

unsigned CompartmentProcess::getLatticeResizeCoord(unsigned aStartCoord)
{
  Comp* aComp(theSpatiocyteStepper->system2Comp(getSuperSystem()));
  aComp->interfaceID = theInterfaceSpecies->getID();
  *theComp = *aComp;
  theVacantSpecies->resetFixedAdjoins();
  theVacantSpecies->setMoleculeRadius(DiffuseRadius);
  if(theLipidSpecies)
    {
      theLipidSpecies->resetFixedAdjoins();
      theLipidSpecies->setMoleculeRadius(LipidRadius);
    }
  //The compartment center point (origin):
  Origin = aComp->centerPoint;
  setCompartmentDimension();
  theComp->dimension = theDimension;
  setLipidCompSpeciesProperties();
  setVacantCompSpeciesProperties();
  subStartCoord = aStartCoord;
  lipStartCoord = aStartCoord+Filaments*Subunits;
  endCoord = lipStartCoord+LipidRows*LipidCols;
  return endCoord-aStartCoord;
}

void CompartmentProcess::setVacantCompSpeciesProperties()
{
  for(unsigned i(0); i != theVacantCompSpecies.size(); ++i)
    {
      theVacantCompSpecies[i]->setDimension(theDimension);
      theVacantCompSpecies[i]->setMoleculeRadius(SubunitRadius);
      theVacantCompSpecies[i]->setDiffuseRadius(DiffuseRadius);
      if(theLipidSpecies)
        {
          theVacantCompSpecies[i]->setMultiscaleVacantSpecies(theLipidSpecies);
        }
    }
}

int CompartmentProcess::getCoefficient(Species* aSpecies)
{
  for(VariableReferenceVector::iterator i(theVariableReferenceVector.begin());
      i != theVariableReferenceVector.end(); ++i)
    {
      if(aSpecies->getVariable() == (*i).getVariable()) 
        {
          return (*i).getCoefficient();
        }
    }
  return 0;
}

Species* CompartmentProcess::coefficient2species(int aCoeff)
{
  for(VariableReferenceVector::iterator i(theVariableReferenceVector.begin());
      i != theVariableReferenceVector.end(); ++i)
    {
      if((*i).getCoefficient() == aCoeff)
        {
          return theSpatiocyteStepper->variable2species((*i).getVariable());
        }
    }
  return NULL;
}

void CompartmentProcess::setLipidCompSpeciesProperties()
{
  for(unsigned i(0); i != theLipidCompSpecies.size(); ++i)
    {
      theLipidCompSpecies[i]->setDimension(theDimension);
      theLipidCompSpecies[i]->setMoleculeRadius(LipidRadius);
    }
}

void CompartmentProcess::updateResizedLattice()
{
  for(unsigned i(0); i != theVacantCompSpecies.size(); ++i)
    {
      //TODO: replace subStartCoord with theVacantSpecies->getCoord(0) 
      //TODO: replace lipStartCoord with theLipidSpecies->getCoord(0) 
      theVacantCompSpecies[i]->setVacStartCoord(subStartCoord, Filaments,
                                                Subunits);
      theVacantCompSpecies[i]->setLipStartCoord(lipStartCoord, LipidRows,
                                                LipidCols);
    }
  for(unsigned i(0); i != theLipidCompSpecies.size(); ++i)
    {
      theLipidCompSpecies[i]->setLipStartCoord(lipStartCoord, LipidRows,
                                                LipidCols);
    }
}

void CompartmentProcess::setSubunitStart()
{
  Point nearest;
  Point farthest;
  getStartVoxelPoint(subunitStart, nearest, farthest);
  double dist(subunitStart.z-nearest.z+nVoxelRadius-nDiffuseRadius);
  subunitStart.z -= int(dist/(nDiffuseRadius*2))*2*nDiffuseRadius;
  dist = subunitStart.y-nearest.y+nVoxelRadius-nDiffuseRadius;
  unsigned cnt(int(dist/(nDiffuseRadius*sqrt(3))));
  subunitStart.y -= cnt*nDiffuseRadius*sqrt(3);
  if(cnt%2 == 1)
    {
      subunitStart.z += nDiffuseRadius;
    }
  if(Autofit)
    {
      Width = (farthest.y-subunitStart.y+nVoxelRadius+nDiffuseRadius)*
        VoxelRadius*2.00001;
      Length = (farthest.z-subunitStart.z+nVoxelRadius+nDiffuseRadius)*
        VoxelRadius*2.00001;
    }
  else
    {
      const double multX(OriginX*theComp->lengthX/2);
      const double multY(OriginY*theComp->lengthY/2);
      const double multZ(OriginZ*theComp->lengthZ/2);
      subunitStart.x += multX;
      subunitStart.y += multY;
      subunitStart.z += multZ;
      Point& center(theComp->centerPoint);
      center.x += multX;
      center.y += multY;
      center.z += multZ;
      rotate(center);
      /*
      subunitStart.y -= (Width/(VoxelRadius*2)-theComp->lengthY)/2;
      subunitStart.z -= (Length/(VoxelRadius*2)-theComp->lengthZ)/2;
      subunitStart.x -= (Height/(VoxelRadius*2)-theComp->lengthX)/2;
      */
    }
}

// y:width:rows
// z:length:cols
void CompartmentProcess::setCompartmentDimension()
{
  setSubunitStart();
  if(Length)
    {
      Subunits = (unsigned)(Length/(DiffuseRadius*2));
    }
  if(Width)
    {
      Filaments = (unsigned)((Width-2*DiffuseRadius)/
                                 (DiffuseRadius*sqrt(3)))+1;
    }
  if(Periodic && Filaments%2 != 0)
    {
      ++Filaments;
    }
  //Need to use 2.00001 here to avoid rounding off error when calculating
  //LipidRows below:
  Width = 2.00001*DiffuseRadius+(Filaments-1)*DiffuseRadius*sqrt(3); 
  Height = 2*DiffuseRadius;
  if(Filaments == 1)
    {
      theDimension = 1;
      Length = Subunits*DiffuseRadius*2;
    }
  else
    {
      theDimension = 2;
      //Add DiffuseRadius for the protrusion from hexagonal arrangement:
      Length = Subunits*DiffuseRadius*2+DiffuseRadius;
    }
  if(theLipidSpecies)
    {
      LipidCols = (unsigned)(Length/(LipidRadius*2));
      LipidRows = (unsigned)((Width-2*LipidRadius)/(LipidRadius*sqrt(3)))+1;
    }
  //TODO: make length, width, height consistent with the definitions
  allocateGrid();
}

void CompartmentProcess::allocateGrid()
{
  //in SpatiocyteStepper:
  //Normalized compartment lengths in terms of lattice voxel radius:
  nParentHeight = theComp->lengthX+nVoxelRadius*2;
  nParentWidth = theComp->lengthY+nVoxelRadius*2;
  nParentLength = theComp->lengthZ+nVoxelRadius*2;
  nLength = Length/(VoxelRadius*2);
  nWidth = Width/(VoxelRadius*2);
  nHeight = Height/(VoxelRadius*2);
  theComp->lengthX = nHeight;
  theComp->lengthY = nWidth;
  theComp->lengthZ = nLength;

  parentOrigin = theComp->centerPoint;
  Point s(parentOrigin);
  parentOrigin.x -= nParentHeight/2;
  parentOrigin.y -= nParentWidth/2;
  parentOrigin.z -= nParentLength/2;
  s = parentOrigin;
  gridCols = (unsigned)ceil(nParentLength/nGridSize);
  gridRows = (unsigned)ceil(nParentWidth/nGridSize);
  gridLayers = (unsigned)ceil(nParentHeight/nGridSize);
  theVacGrid.resize(gridCols*gridRows*gridLayers);
  /*
  if(theLipidSpecies)
    {
      theLipGrid.resize(gridCols*gridRows*gridLayers);
    }
    */
  //Actual surface area = Width*Length
  theComp->actualArea = Width*Length;
}

void CompartmentProcess::initializeThird()
{
  if(!isCompartmentalized)
    {
      thePoints.resize(endCoord-subStartCoord);
      vacStartIndex = theVacantSpecies->size();
      intStartIndex = theInterfaceSpecies->size();
      initializeVectors();
      initializeFilaments(subunitStart, Filaments, Subunits, nDiffuseRadius,
                          theVacantSpecies, subStartCoord);
      elongateFilaments(theVacantSpecies, subStartCoord, Filaments, Subunits,
                        nDiffuseRadius);
      connectFilaments(subStartCoord, Filaments, Subunits);
      setGrid(theVacantSpecies, theVacGrid, subStartCoord);
      interfaceSubunits();
      initializeFilaments(lipidStart, LipidRows, LipidCols, nLipidRadius,
                          theLipidSpecies, lipStartCoord);
      elongateFilaments(theLipidSpecies, lipStartCoord, LipidRows,
                        LipidCols, nLipidRadius);
      connectFilaments(lipStartCoord, LipidRows, LipidCols);
      setDiffuseSize(lipStartCoord, endCoord);
      //setGrid(theLipidSpecies, theLipGrid, lipStartCoord);
      setSpeciesIntersectLipids();
      isCompartmentalized = true;
    }
  theVacantSpecies->setIsPopulated();
  if(theLipidSpecies)
    {
      theLipidSpecies->setIsPopulated();
    }
  theInterfaceSpecies->setIsPopulated();
}

void CompartmentProcess::setGrid(Species* aSpecies,
                                 std::vector<std::vector<unsigned> >& aGrid,
                                 unsigned aStartCoord)
{ 
  if(aSpecies)
    {
      for(unsigned i(vacStartIndex); i != aSpecies->size(); ++i)
        {
          Point& aPoint(*(*theLattice)[aStartCoord+i-vacStartIndex].point);
          const int row((int)((aPoint.y-parentOrigin.y)/nGridSize));
          const int col((int)((aPoint.z-parentOrigin.z)/nGridSize));
          const int layer((int)((aPoint.x-parentOrigin.x)/nGridSize));
          if(row >= 0 && row < gridRows && layer >= 0 && layer < gridLayers &&
             col >= 0 && col < gridCols)
            {
              aGrid[row+
                gridRows*layer+
                gridRows*gridLayers*col].push_back(aStartCoord+i-vacStartIndex);
            }
        }
    }
}

// y:width:rows:filaments
// z:length:cols:subunits
void CompartmentProcess::setSpeciesIntersectLipids()
{
  setAdjoinOffsets();
  if(RegularLattice)
    {
      for(unsigned i(0); i != theLipidCompSpecies.size(); ++i)
        {
          theLipidCompSpecies[i]->setAdjoinOffsets(theAdjoinOffsets,
                                                   theRowOffsets);
        }
      for(unsigned i(0); i != theVacantCompSpecies.size(); ++i)
        {
          theVacantCompSpecies[i]->setIntersectOffsets(theAdjoinOffsets,
                                                       theRowOffsets,
                                                       theLipidSpecies,
                                                       lipidStart, Filaments,
                                                       Subunits, nLipidRadius,
                                                       SubunitAngle,
                                                       surfaceNormal);
        }
    }
  else
    {
      for(unsigned i(0); i != theVacantCompSpecies.size(); ++i)
        {
          if(theVacantCompSpecies[i]->getIsMultiscale())
            {
              theVacantCompSpecies[i]->setIntersectLipids(theLipidSpecies,
                                                lipidStart, nGridSize, gridCols,
                                                gridRows, theVacGrid, Filaments,
                                                Subunits);
            }
        }
    }
}

void CompartmentProcess::getStartVoxelPoint(Point& start, Point& nearest,
                                            Point& farthest)
{
  Comp* aComp(theSpatiocyteStepper->system2Comp(getSuperSystem()));
  Species* surface(aComp->surfaceSub->vacantSpecies);
  double dist(0);
  Point origin;
  origin.x = 0;
  origin.y = 0;
  origin.z = 0;
  if(surface->size())
    {
      nearest = theSpatiocyteStepper->coord2point(surface->getCoord(0));
      farthest = nearest; 
      origin.x = nearest.x;
      dist = distance(nearest, origin);
      start = nearest;
    }
  for(unsigned i(1); i < surface->size(); ++i)
    {
      Point aPoint(theSpatiocyteStepper->coord2point(surface->getCoord(i)));
      if(aPoint.x < nearest.x)
        {
          nearest.x = aPoint.x;
          origin.x = aPoint.x;
          dist = distance(aPoint, origin);
          start = nearest;
        }
      else if(aPoint.x == nearest.x)
        {
          origin.x = aPoint.x;
          double aDist(distance(aPoint, origin));
          if(aDist < dist)
            {
              dist = aDist;
              start = aPoint;
            }
        }
      if(aPoint.y < nearest.y)
        {
          nearest.y = aPoint.y;
        }
      if(aPoint.z < nearest.z)
        {
          nearest.z = aPoint.z;
        }
      if(aPoint.x > farthest.x)
        {
          farthest.x = aPoint.x;
        }
      if(aPoint.y > farthest.y)
        {
          farthest.y = aPoint.y;
        }
      if(aPoint.z > farthest.z)
        {
          farthest.z = aPoint.z;
        }
    }
}

void CompartmentProcess::initializeVectors()
{
  lengthStart = subunitStart;
  //For Lipid start:
  lengthStart.z -= nDiffuseRadius;
  lengthStart.y -= nDiffuseRadius;
  rotate(subunitStart);
  rotate(lengthStart);

  lengthVector.x = 0;
  lengthVector.y = 0;
  lengthVector.z = 1;
  rotate(lengthVector);
  lengthEnd = disp(lengthStart, lengthVector, nLength);

  widthVector.x = 0;
  widthVector.y = 1;
  widthVector.x = 0;
  rotate(widthVector);
  widthEnd = disp(lengthEnd, widthVector, nWidth);

  heightVector.x = 1;
  heightVector.y = 0;
  heightVector.z = 0;
  rotate(heightVector);
  heightEnd = disp(widthEnd, heightVector, nHeight);

  if(theLipidSpecies)
    {
      lipidStart = lengthStart;
      disp_(lipidStart, lengthVector, nLipidRadius);
      disp_(lipidStart, widthVector, nLipidRadius);
    }

  Point center(lengthStart);
  disp_(center, lengthVector, nLength/2);
  disp_(center, widthVector, nWidth/2);
  theComp->centerPoint = center;

  //Set up surface vectors:
  surfaceNormal = cross(lengthVector, widthVector);
  surfaceNormal = norm(surfaceNormal);
  surfaceDisplace = dot(surfaceNormal, widthEnd);
  lengthDisplace = dot(lengthVector, lengthStart);
  lengthDisplaceOpp = dot(lengthVector, lengthEnd);
  widthDisplace = dot(widthVector, lengthEnd);
  widthDisplaceOpp = dot(widthVector, widthEnd);
}

void CompartmentProcess::rotate(Point& V)
{
  if(!Autofit)
    {
      theSpatiocyteStepper->rotateX(RotateX, &V, -1);
      theSpatiocyteStepper->rotateY(RotateY, &V, -1);
      theSpatiocyteStepper->rotateZ(RotateZ, &V, -1);
    }
}

void CompartmentProcess::initializeFilaments(Point& aStartPoint, unsigned aRows,
                                             unsigned aCols, double aRadius,
                                             Species* aVacant,
                                             unsigned aStartCoord)
{
  //The first comp voxel must have the aStartCoord:
  if(aStartCoord != endCoord)
    {
      addCompVoxel(0, 0, aStartPoint, aVacant, aStartCoord, aCols);
    }
  for(unsigned i(1); i < aRows; ++i)
    {
      Point U(aStartPoint);
      disp_(U, widthVector, i*aRadius*sqrt(3)); 
      if(i%2 == 1)
        {
          disp_(U, lengthVector, -aRadius); 
        }
      addCompVoxel(i, 0, U, aVacant, aStartCoord, aCols);
    }
}

Voxel* CompartmentProcess::addCompVoxel(unsigned rowIndex, 
                                        unsigned colIndex,
                                        Point& aPoint,
                                        Species* aVacant,
                                        unsigned aStartCoord,
                                        unsigned aCols)
{
  const unsigned aCoord(aStartCoord+rowIndex*aCols+colIndex);
  Voxel& aVoxel((*theLattice)[aCoord]);
  aVoxel.point = &thePoints[aStartCoord-subStartCoord+rowIndex*aCols+colIndex];
  *aVoxel.point = aPoint;
  if(RegularLattice)
    {
      aVoxel.adjoiningSize = theDiffuseSize;
      aVoxel.diffuseSize = theDiffuseSize;
      aVoxel.adjoiningCoords = new unsigned int[theDiffuseSize];
      for(unsigned i(0); i != theDiffuseSize; ++i)
        {
          aVoxel.adjoiningCoords[i] = theNullCoord;
        }
    }
  else
    {
      aVoxel.adjoiningSize = 0;
    }
  aVacant->addCompVoxel(aCoord);
  return &aVoxel;
}

void CompartmentProcess::elongateFilaments(Species* aVacant,
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
          addCompVoxel(i, j, A, aVacant, aStartCoord, aCols);
        }
    }
}


void CompartmentProcess::connectSubunit(unsigned a, unsigned b, 
                                        unsigned adjoinA, unsigned adjoinB)
{
  Voxel& voxelA((*theLattice)[a]);
  Voxel& voxelB((*theLattice)[b]);
  if(RegularLattice)
    {
      voxelA.adjoiningCoords[adjoinA] = b;
      voxelB.adjoiningCoords[adjoinB] = a;
    }
  else
    {
      addAdjoin(voxelA, b);
      addAdjoin(voxelB, a);
    }
}

/*
 row0   row1    row2
 fil0   fil1    fil2
 [NW] [ NORTH ] [NE] sub0, col0
      [subunit]      sub1, col1
 [SW] [ SOUTH ] [SE] sub2, col2
*/

void CompartmentProcess::setAdjoinOffsets()
{
  theAdjoinOffsets.resize(2);
  theAdjoinOffsets[0].resize(theDiffuseSize);
  theAdjoinOffsets[0][NORTH] = -1;
  theAdjoinOffsets[0][SOUTH] = 1;
  theAdjoinOffsets[0][NW] = -LipidCols;
  theAdjoinOffsets[0][SW] = -LipidCols+1;
  theAdjoinOffsets[0][NE] = LipidCols;
  theAdjoinOffsets[0][SE] = LipidCols+1;

  theAdjoinOffsets[1].resize(theDiffuseSize);
  theAdjoinOffsets[1][NORTH] = -1;
  theAdjoinOffsets[1][SOUTH] = 1;
  theAdjoinOffsets[1][NW] = -LipidCols-1;
  theAdjoinOffsets[1][SW] = -LipidCols;
  theAdjoinOffsets[1][NE] = LipidCols-1;
  theAdjoinOffsets[1][SE] = LipidCols;

  theRowOffsets.resize(theDiffuseSize);
  theRowOffsets[NORTH] = 0;
  theRowOffsets[SOUTH] = 0;
  theRowOffsets[NW] = -1;
  theRowOffsets[SW] = -1;
  theRowOffsets[NE] = 1;
  theRowOffsets[SE] = 1;
}

// y:width:rows:filaments
// z:length:cols:subunits
void CompartmentProcess::connectFilaments(unsigned aStartCoord,
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
              if(j%2 == 1)
                {
                  if(j+1 < aRows)
                    {
                      //periodic NE-SW 
                      b = aStartCoord+(j+1)*aCols+aCols-1; 
                      connectSubunit(a, b, NE, SW);
                    }
                  else if(j == aRows-1)
                    {
                      //periodic NE-SW 
                      b = aStartCoord+aCols-1; 
                      connectSubunit(a, b, NE, SW);
                    }
                  //periodic NW-SE
                  b = aStartCoord+(j-1)*aCols+aCols-1; 
                  connectSubunit(a, b, NW, SE);
                }
            }
          if(j > 0)
            {
              if(j%2 == 1)
                {
                  //SW-NE
                  unsigned a(aStartCoord+j*aCols+i);
                  unsigned b(aStartCoord+(j-1)*aCols+i); 
                  connectSubunit(a, b, SW, NE);
                  if(i > 0)
                    {
                      //NW-SE
                      b = aStartCoord+(j-1)*aCols+(i-1); 
                      connectSubunit(a, b, NW, SE);
                    }
                }
              else
                {
                  //NW-SE
                  unsigned a(aStartCoord+j*aCols+i);
                  unsigned b(aStartCoord+(j-1)*aCols+i); 
                  connectSubunit(a, b, NW, SE);
                  if(i+1 < aCols)
                    {
                      //SW-NE
                      b = aStartCoord+(j-1)*aCols+(i+1); 
                      connectSubunit(a, b, SW, NE);
                    }
                }
            }
        }
      if(Periodic && aRows > 1)
        { 
          //periodic NW-SE
          unsigned a(aStartCoord+i); //row 0
          unsigned b(aStartCoord+(aRows-1)*aCols+i); 
          connectSubunit(a, b, NW, SE);
          if(i+1 < aCols)
            {
              //periodic SW-NE
              b = aStartCoord+(aRows-1)*aCols+(i+1); 
              connectSubunit(a, b, SW, NE);
            }
        }
    }
}

void CompartmentProcess::setDiffuseSize(unsigned start, unsigned end)
{
  for(unsigned i(start); i != end; ++i)
    {
      Voxel& subunit((*theLattice)[i]);
      subunit.diffuseSize = subunit.adjoiningSize;
    }
}

void CompartmentProcess::addAdjoin(Voxel& aVoxel, unsigned coord)
{
  unsigned* temp(new unsigned[aVoxel.adjoiningSize+1]);
  for(unsigned int i(0); i != aVoxel.adjoiningSize; ++i)
    {
      //Avoid duplicated adjoins:
      if(aVoxel.adjoiningCoords[i] == coord)
        {
          delete[] temp;
          return;
        }
      temp[i] = aVoxel.adjoiningCoords[i];
    }
  delete[] aVoxel.adjoiningCoords;
  temp[aVoxel.adjoiningSize++] = coord;
  aVoxel.adjoiningCoords = temp;
}

void CompartmentProcess::interfaceSubunits()
{
  enlistSubunitIntersectInterfaceVoxels();
  enlistPlaneIntersectInterfaceVoxels();
  enlistOrphanSubunitInterfaceVoxels();
  setDiffuseSize(subStartCoord, lipStartCoord);
  enlistSubunitInterfaceAdjoins();
}

void CompartmentProcess::enlistOrphanSubunitInterfaceVoxels()
{
  //No need to enlist intersecting interface voxel again since
  //it would have been done for all already if the 
  //nDiffuseRadius > nVoxelRadius:
  if(nDiffuseRadius > nVoxelRadius)
    {
      return;
    }
  for(unsigned i(subStartCoord); i != lipStartCoord; ++i)
    {
      if(!subunitInterfaces[i-subStartCoord].size())
        {
          setSubunitInterfaceVoxels(i, 0.75*(nDiffuseRadius+nVoxelRadius)); 
          if(!subunitInterfaces[i-subStartCoord].size())
            {
              setSubunitInterfaceVoxels(i, nDiffuseRadius+nVoxelRadius); 
              if(!subunitInterfaces[i-subStartCoord].size())
                {
                  cout << getPropertyInterface().getClassName() << ":"
                    << getFullID().asString() << 
                    ": subunit is still orphaned" << std::endl;
                }
            }
        }
    }
}

bool CompartmentProcess::setSubunitInterfaceVoxels(const unsigned i,
                                                   const double aDist,
                                                   const bool isSingle)
{
  Voxel& subunit((*theLattice)[i]);
  Point center(*subunit.point);
  Point bottomLeft(*subunit.point);
  Point topRight(*subunit.point);
  bottomLeft.x -= nDiffuseRadius+theSpatiocyteStepper->getColLength();
  bottomLeft.y -= nDiffuseRadius+theSpatiocyteStepper->getLayerLength();
  bottomLeft.z -= nDiffuseRadius+theSpatiocyteStepper->getRowLength();
  topRight.x += nDiffuseRadius+theSpatiocyteStepper->getColLength();
  topRight.y += nDiffuseRadius+theSpatiocyteStepper->getLayerLength();
  topRight.z += nDiffuseRadius+theSpatiocyteStepper->getRowLength();
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
              //Return if we have enlisted at least on interface voxel
              //in the case of subunits are equal or smaller than
              //voxels:
              addInterfaceVoxel(i, m, aDist);
              if(isSingle && theInterfaceSpecies->size() > intStartIndex)
                {
                  return true;
                }
            }
        }
    }
  return false;
}

void CompartmentProcess::enlistSubunitIntersectInterfaceVoxels()
{
  subunitInterfaces.resize(Filaments*Subunits);
  for(unsigned i(subStartCoord); i != lipStartCoord; ++i)
    {
      //If nDiffuseRadius <= nVoxelRadius, just enlist one interface voxel:
      if(setSubunitInterfaceVoxels(i, std::max(nDiffuseRadius, nVoxelRadius),
                                   nDiffuseRadius <= nVoxelRadius))
        {
          return;
        }
    }
}

void CompartmentProcess::addInterfaceVoxel(unsigned subunitCoord,
                                           unsigned voxelCoord,
                                           const double aDist)
{ 
  Voxel& subunit((*theLattice)[subunitCoord]);
  Point subunitPoint(*subunit.point);
  Point voxelPoint(theSpatiocyteStepper->coord2point(voxelCoord));
  double dist(distance(subunitPoint, voxelPoint));
  //Should use SubunitRadius instead of DiffuseRadius since it is the
  //actual size of the subunit. Nope, the distance is too far when using
  //SubunitRadius:
  if(dist < aDist)
    {
      Voxel& voxel((*theLattice)[voxelCoord]);
      //theSpecies[6]->addMolecule(&voxel);
      //Insert voxel in the list of interface voxels if was not already:
      //if(voxel.id == theComp->vacantSpecies->getID())
      //if(getID(voxel) != theInterfaceSpecies->getID())
      if(theSpecies[getID(voxel)]->getIsCompVacant())
        {
          //theSpecies[5]->addMolecule(&voxel);
          theInterfaceSpecies->addMolecule(&voxel);
        }
      //each subunit list has unique (no duplicates) interface voxels:
      subunitInterfaces[subunitCoord-subStartCoord].push_back(voxelCoord);
    }
}

void CompartmentProcess::addInterfaceVoxel(Voxel& aVoxel, Point& aPoint)
{ 
  theInterfaceSpecies->addMolecule(&aVoxel);
  const int row((int)((aPoint.y-parentOrigin.y)/nGridSize));
  const int col((int)((aPoint.z-parentOrigin.z)/nGridSize));
  const int layer((int)((aPoint.x-parentOrigin.x)/nGridSize));
  if(row >= 0 && row < gridRows && layer >= 0 && layer < gridLayers &&
     col >= 0 && col < gridCols)
    {
      for(int i(std::max(0, layer-1)); i != std::min(layer+2, gridLayers); ++i)
        {
          for(int j(std::max(0, col-1)); j != std::min(col+2, gridCols); ++j)
            {
              for(int k(std::max(0, row-1)); k != std::min(row+2, gridRows);
                  ++k)
                {
                  const std::vector<unsigned>& coords(theVacGrid[k+gridRows*i+
                                                      gridRows*gridLayers*j]);
                  for(unsigned l(0); l != coords.size(); ++l)
                    {
                      const unsigned subCoord(coords[l]);
                      const Point& subPoint(*(*theLattice)[subCoord].point);
                      const double dist(distance(subPoint, aPoint));
                      if(dist < nDiffuseRadius+nVoxelRadius)
                        {
                          subunitInterfaces[subCoord-subStartCoord].push_back(
                                                              aVoxel.coord);
                        }
                    }
                }
            }
        }
    }
}

void CompartmentProcess::enlistSubunitInterfaceAdjoins()
{
  for(unsigned i(0); i != subunitInterfaces.size(); ++i)
    {
      for(unsigned j(0); j != subunitInterfaces[i].size(); ++j)
        {
          Voxel& subunit((*theLattice)[i+subStartCoord]);
          Voxel& interface((*theLattice)[subunitInterfaces[i][j]]);
          addAdjoin(interface, i+subStartCoord);
          for(unsigned k(0); k != interface.diffuseSize; ++k)
            {
              unsigned coord(interface.adjoiningCoords[k]);
              Voxel& adjoin((*theLattice)[coord]);
              //if(getID(adjoin) != theInterfaceSpecies->getID())
              if(theSpecies[getID(adjoin)]->getIsCompVacant() &&
                 isCorrectSide(adjoin.coord))
                {
                  addAdjoin(subunit, coord);
                }
            }
        }
    }
}

bool CompartmentProcess::isCorrectSide(const unsigned aCoord)
{
  Point aPoint(theSpatiocyteStepper->coord2point(aCoord));
  switch(SurfaceDirection)
    {
      //Is inside
    case 0:
      if(isOnAboveSurface(aPoint))
        {
          return true;
        }
      return false;
      //Is outside
    case 1:
      if(!isOnAboveSurface(aPoint))
        {
          return true;
        }
      return false;
      //Is bidirectional
    default:
      return true;
    }
  return true;
}

void CompartmentProcess::enlistPlaneIntersectInterfaceVoxels()
{
  for(unsigned i(intStartIndex); i != theInterfaceSpecies->size(); ++i)
    {
      unsigned voxelCoord(theInterfaceSpecies->getCoord(i));
      Voxel& anInterface((*theLattice)[voxelCoord]);
      for(unsigned j(0); j != theAdjoiningCoordSize; ++j)
        {
          Voxel& adjoin((*theLattice)[anInterface.adjoiningCoords[j]]);
          //if(getID(adjoin) != theInterfaceSpecies->getID())
          if(theSpecies[getID(adjoin)]->getIsCompVacant())
            {
              Point aPoint(theSpatiocyteStepper->coord2point(adjoin.coord));
              if(isInside(aPoint))
                {
                  addPlaneIntersectInterfaceVoxel(adjoin, aPoint);
                }
            }
        }
    }
}

bool CompartmentProcess::isInside(Point& aPoint)
{
  double disp(point2planeDisp(aPoint, lengthVector, lengthDisplace));
  if(disp >= 0)
    {
      disp = point2planeDisp(aPoint, lengthVector, lengthDisplaceOpp);
      if(disp <= 0)
        {
          disp = point2planeDisp(aPoint, widthVector, widthDisplaceOpp);
          if(disp <= 0)
            {
              disp = point2planeDisp(aPoint, widthVector, widthDisplace);
              if(disp >= 0)
                {
                  return true;
                }
            }
        }
    }
  return false;
}

bool CompartmentProcess::isOnAboveSurface(Point& aPoint)
{
  double disp(point2planeDisp(aPoint, surfaceNormal, surfaceDisplace));
  if(disp >= 0)
    {
      return true;
    }
  return false;
}

void CompartmentProcess::addPlaneIntersectInterfaceVoxel(Voxel& aVoxel,
                                                         Point& aPoint)
{
  double dispA(point2planeDisp(aPoint, surfaceNormal, surfaceDisplace));
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      Voxel& adjoin((*theLattice)[aVoxel.adjoiningCoords[i]]);
      //if(getID(adjoin) != theInterfaceSpecies->getID())
      if(theSpecies[getID(adjoin)]->getIsCompVacant())
        {
          Point pointB(theSpatiocyteStepper->coord2point(adjoin.coord));
          double dispB(point2planeDisp(pointB, surfaceNormal, surfaceDisplace));
          //if not on the same side of the plane, or one of it is on the plane
          //and the other is not:
          if((dispA < 0) != (dispB < 0))
            {
              //If the voxel is nearer to the plane:
              if(abs(dispA) < abs(dispB))
                { 
                  addInterfaceVoxel(aVoxel, aPoint);
                  return;
                }
              //If the adjoin is nearer to the plane:
              else
                {
                  addInterfaceVoxel(adjoin, pointB);
                }
            }
        }
    }
}

void CompartmentProcess::printParameters()
{
  cout << getPropertyInterface().getClassName() << "[" <<
    getFullID().asString() << "]" << std::endl;
  cout << "  width:" << Width << " length:" << Length <<
    " area:" << Width*Length << " LipidRows:" << LipidRows << " LipidCols:" <<
    LipidCols << " Filaments:" << Filaments << " Subunits:" << Subunits <<
    std::endl;
  if(theLipidSpecies)
    {
      cout << "  " << getIDString(theLipidSpecies) << 
        " number:" << theLipidSpecies->size() << std::endl;
      for(unsigned i(0); i != theLipidCompSpecies.size(); ++i)
        {
          cout << "    " << getIDString(theLipidCompSpecies[i]) <<
            " number:" << theLipidCompSpecies[i]->size() << std::endl;
        }
    } 
  cout << "  " << getIDString(theVacantSpecies) << 
    " number:" << theVacantSpecies->size() << std::endl;
      for(unsigned i(0); i != theVacantCompSpecies.size(); ++i)
        {
          cout << "    " << getIDString(theVacantCompSpecies[i]) <<
            " number:" << theVacantCompSpecies[i]->size() << std::endl;
        }
}
