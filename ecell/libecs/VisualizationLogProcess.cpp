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

#include "VisualizationLogProcess.hpp"

LIBECS_DM_INIT(VisualizationLogProcess, Process); 

void VisualizationLogProcess::initializeLog()
{
  unsigned int aLatticeType(theSpatiocyteStepper->getLatticeType());
  theLogFile.write((char*)(&aLatticeType), sizeof(aLatticeType));
  theLogFile.write((char*)(&theMeanCount), sizeof(theMeanCount));
  unsigned int aStartCoord(0);
  theLogFile.write((char*)(&aStartCoord), sizeof(aStartCoord));
  unsigned int aRowSize(theSpatiocyteStepper->getRowSize());
  theLogFile.write((char*)(&aRowSize), sizeof(aRowSize));
  unsigned int aLayerSize(theSpatiocyteStepper->getLayerSize());
  theLogFile.write((char*)(&aLayerSize), sizeof(aLayerSize));
  unsigned int aColSize(theSpatiocyteStepper->getColSize());
  theLogFile.write((char*)(&aColSize), sizeof(aColSize));
  Point aCenterPoint(theSpatiocyteStepper->getCenterPoint());
  double aRealRowSize(aCenterPoint.z*2);
  theLogFile.write((char*)(&aRealRowSize), sizeof(aRealRowSize));
  double aRealLayerSize(aCenterPoint.y*2);
  theLogFile.write((char*)(&aRealLayerSize), sizeof(aRealLayerSize));
  double aRealColSize(aCenterPoint.x*2);
  theLogFile.write((char*)(&aRealColSize), sizeof(aRealColSize));
  unsigned int aLatticeSpSize(theLatticeSpecies.size());
  theLogFile.write((char*)(&aLatticeSpSize), sizeof(aLatticeSpSize));
  unsigned int aPolymerSize(thePolymerSpecies.size());
  theLogFile.write((char*)(&aPolymerSize), sizeof(aPolymerSize));
  unsigned int aReservedSize(0);
  theLogFile.write((char*)(&aReservedSize), sizeof(aReservedSize));
  unsigned int anOffLatticeSpSize(theOffLatticeSpecies.size());
  theLogFile.write((char*)(&anOffLatticeSpSize), sizeof(anOffLatticeSpSize));
  //theLogMarker is a constant throughout the simulation:
  theLogFile.write((char*)(&theLogMarker), sizeof(theLogMarker));
  double aVoxelRadius(theSpatiocyteStepper->getVoxelRadius());
  theLogFile.write((char*)(&aVoxelRadius), sizeof(aVoxelRadius));
  for(unsigned int i(0); i != theLatticeSpecies.size(); ++i)
    {
      unsigned int aStringSize(
       theLatticeSpecies[i]->getVariable()->getFullID().asString().size());
      theLogFile.write((char*)(&aStringSize), sizeof(aStringSize));
      theLogFile.write(
       theLatticeSpecies[i]->getVariable()->getFullID().asString().c_str(),
       aStringSize);
      double aRadius(theLatticeSpecies[i]->getRadius());
      theLogFile.write((char*)(&aRadius), sizeof(aRadius));
    }
  for(unsigned int i(0); i!=thePolymerSpecies.size(); ++i)
    {
      unsigned int aPolymerIndex(thePolymerIndex[i]);
      theLogFile.write((char*) (&aPolymerIndex), sizeof(aPolymerIndex));
      double aRadius(thePolymerSpecies[i]->getRadius());
      theLogFile.write((char*)(&aRadius), sizeof(aRadius));
    }
  for(unsigned int i(0); i != theOffLatticeSpecies.size(); ++i)
    {
      unsigned int aStringSize(
       theOffLatticeSpecies[i]->getVariable()->getFullID().asString().size());
      theLogFile.write((char*)(&aStringSize), sizeof(aStringSize));
      theLogFile.write(
       theOffLatticeSpecies[i]->getVariable()->getFullID().asString().c_str(),
       aStringSize);
      double aRadius(theOffLatticeSpecies[i]->getRadius());
      theLogFile.write((char*)(&aRadius), sizeof(aRadius));
    }
}

void VisualizationLogProcess::logMolecules(int anIndex)
{
  Species* aSpecies(theLatticeSpecies[anIndex]);
  //No need to log lipid or non diffusing vacant molecules since we have
  //already logged them once during initialization:
  if(aSpecies->getIsCompVacant())
    {
      return;
    }
  //The remaining vacant molecules must be diffusive, so we need to update
  //them before logging their position:
  //Also update the molecules of the tag species:
  aSpecies->updateMolecules();
  theLogFile.write((char*)(&anIndex), sizeof(anIndex));
  //The species molecule size:
  int aSize(aSpecies->size());
  theLogFile.write((char*)(&aSize), sizeof(aSize)); 
  for(int i(0); i != aSize; ++i)
    {
      unsigned int aCoord(aSpecies->getCoord(i));
      theLogFile.write((char*)(&aCoord), sizeof(aCoord));
    }
}  

void VisualizationLogProcess::logOffLattice(int anIndex)
{
  Species* aSpecies(theOffLatticeSpecies[anIndex]);
  if(aSpecies->getIsVacant() && !aSpecies->getIsDiffusiveVacant() &&
     !aSpecies->getIsReactiveVacant())
    {
      return;
    }
  aSpecies->updateMolecules();
  theLogFile.write((char*)(&anIndex), sizeof(anIndex));
  //The species molecule size:
  int aSize(aSpecies->size());
  theLogFile.write((char*)(&aSize), sizeof(aSize)); 
  for(int i(0); i != aSize; ++i)
    {
      Point aPoint(aSpecies->getPoint(i));
      theLogFile.write((char*)(&aPoint), sizeof(aPoint));
    }
}  

void VisualizationLogProcess::logPolymers(int anIndex)
{
  Species* aSpecies(thePolymerSpecies[anIndex]);
  theLogFile.write((char*)(&anIndex), sizeof(anIndex));
  //The species molecule size:
  int aSize(aSpecies->size());
  theLogFile.write((char*)(&aSize), sizeof(aSize)); 
  for(int i(0); i != aSize; ++i)
    {
      Point aPoint(aSpecies->getPoint(i));
      theLogFile.write((char*)(&aPoint), sizeof(aPoint));
    }
}  

void VisualizationLogProcess::logSourceMolecules(int anIndex)
{
  Species* aSpecies(thePolymerSpecies[anIndex]);
  int aSourceIndex(theLatticeSpecies.size()+anIndex);
  theLogFile.write((char*)(&aSourceIndex), sizeof(aSourceIndex));
  const std::vector<unsigned int> aCoords(aSpecies->getSourceCoords());
  int aSize(aCoords.size());
  theLogFile.write((char*)(&aSize), sizeof(aSize)); 
  for(unsigned int i(0); i != aCoords.size(); ++i)
    {
      unsigned int aCoord(aCoords[i]);
      theLogFile.write((char*)(&aCoord), sizeof(aCoord));
    }
}  

void VisualizationLogProcess::logTargetMolecules(int anIndex)
{
  Species* aSpecies(thePolymerSpecies[anIndex]);
  int aTargetIndex(theLatticeSpecies.size()+thePolymerSpecies.size()+anIndex);
  theLogFile.write((char*)(&aTargetIndex), sizeof(aTargetIndex));
  const std::vector<unsigned int> aCoords(aSpecies->getTargetCoords());
  int aSize(aCoords.size());
  theLogFile.write((char*)(&aSize), sizeof(aSize)); 
  for(unsigned int i(0); i != aCoords.size(); ++i)
    {
      unsigned int aCoord(aCoords[i]);
      theLogFile.write((char*)(&aCoord), sizeof(aCoord));
    }
}  

void VisualizationLogProcess::logSharedMolecules(int anIndex)
{
  Species* aSpecies(thePolymerSpecies[anIndex]);
  int aSharedIndex(theLatticeSpecies.size()+thePolymerSpecies.size()*2+anIndex);
  theLogFile.write((char*)(&aSharedIndex), sizeof(aSharedIndex));
  const std::vector<unsigned int> aCoords(aSpecies->getSharedCoords());
  int aSize(aCoords.size());
  theLogFile.write((char*)(&aSize), sizeof(aSize)); 
  for(unsigned int i(0); i != aCoords.size(); ++i)
    {
      unsigned int aCoord(aCoords[i]);
      theLogFile.write((char*)(&aCoord), sizeof(aCoord));
    }
}  

void VisualizationLogProcess::logSpecies()
{
  double aCurrentTime(theSpatiocyteStepper->getCurrentTime());
  theLogFile.write((char*)(&aCurrentTime), sizeof(aCurrentTime));
  for(unsigned int i(0); i != theLatticeSpecies.size(); ++i)
    {
      logMolecules(i);
    }
  for(unsigned int i(0); i != thePolymerSpecies.size(); ++i)
    {
      logSourceMolecules(i);
    }
  for(unsigned int i(0); i != thePolymerSpecies.size(); ++i)
    {
      logTargetMolecules(i);
    }
  for(unsigned int i(0); i != thePolymerSpecies.size(); ++i)
    {
      logSharedMolecules(i);
    }
  /*
  for(unsigned int i(0); i!=theReservedSpecies.size(); ++i)
    {
      logReservedMolecules(i);
    }
    */
  //theLogMarker is a constant throughout the simulation:
  theLogFile.write((char*)(&theLogMarker), sizeof(theLogMarker));
  for(unsigned int i(0); i != thePolymerSpecies.size(); ++i)
    {
      logPolymers(i);
    }
  for(unsigned int i(0); i != theOffLatticeSpecies.size(); ++i)
    {
      logOffLattice(i);
    }
  //theLogMarker is a constant throughout the simulation:
  theLogFile.write((char*)(&theLogMarker), sizeof(theLogMarker));
}

void VisualizationLogProcess::logCompVacant()
{
  double aCurrentTime(theSpatiocyteStepper->getCurrentTime());
  theLogFile.write((char*)(&aCurrentTime), sizeof(aCurrentTime));
  for(unsigned int i(0); i != theLatticeSpecies.size(); ++i)
    {
      if(theLatticeSpecies[i]->getIsCompVacant())
        {
          Species* aVacantSpecies(theLatticeSpecies[i]);
          //The species index in the process:
          theLogFile.write((char*)(&i), sizeof(i));
          //The species molecule size:
          unsigned int aSize(aVacantSpecies->size());
          theLogFile.write((char*)(&aSize), sizeof(aSize)); 
          for(unsigned int j(0); j != aSize; ++j)
            {
              unsigned int aCoord(aVacantSpecies->getCoord(j));
              theLogFile.write((char*)(&aCoord), sizeof(aCoord));
            }  
        }
    }
  //theLogMarker is a constant throughout the simulation:
  theLogFile.write((char*)(&theLogMarker), sizeof(theLogMarker));
  for(unsigned int i(0); i != theOffLatticeSpecies.size(); ++i)
    {
      if(theOffLatticeSpecies[i]->getIsCompVacant())
        {
          Species* aSpecies(theOffLatticeSpecies[i]);
          theLogFile.write((char*)(&i), sizeof(i));
          //The species molecule size:
          int aSize(aSpecies->size());
          theLogFile.write((char*)(&aSize), sizeof(aSize)); 
          for(int i(0); i != aSize; ++i)
            {
              Point aPoint(aSpecies->getPoint(i));
              theLogFile.write((char*)(&aPoint), sizeof(aPoint));
            }
        }
    }
  theLogFile.write((char*)(&theLogMarker), sizeof(theLogMarker));
}
