//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2006-2009 Keio University
//
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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

#ifdef WITH_NO_DUMP
#include <sstream>
#endif

#include <time.h>
#include <gsl/gsl_randist.h>
#include <libecs/Model.hpp>
#include <libecs/System.hpp>
#include <libecs/Process.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/VariableReference.hpp>
#include "SpatiocyteStepper.hpp"
#include "SpatiocyteSpecies.hpp"
#include "SpatiocyteProcessInterface.hpp"
#include "ReactionProcessInterface.hpp"


#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "Model.hpp"

#include "SpatiocyteStepper.hpp"

namespace libecs
{

LIBECS_DM_INIT_STATIC(SpatiocyteStepper, Stepper);

void SpatiocyteStepper::initialize()
{
  if(isInitialized)
    {
      return;
    }
  isInitialized = true;
  Stepper::initialize(); 
  if(theProcessVector.empty())
    {
      THROW_EXCEPTION(InitializationFailed,
                      getPropertyInterface().getClassName() + 
                      ": at least one Process must be defined in this" +
                      " Stepper.");
    } 

  {
#ifndef WITH_NO_DUMP
  using namespace std;
#else
  std::stringstream cout;
#endif

  cout << "1. checking model..." << std::endl;
  checkModel();
  //We need a Comp tree to assign the voxels to each Comp
  //and get the available number of vacant voxels. The compartmentalized
  //vacant voxels are needed to randomly place molecules according to the
  //Comp:
  cout << "2. creating compartments..." << std::endl;
  registerComps();
  setCompsProperties();
  cout << "3. setting up lattice properties..." << std::endl;
  setLatticeProperties(); 
  broadcastLatticeProperties();
  setCompsCenterPoint();
  //All species have been created at this point, we initialize them now:
  cout << "4. initializing species..." << std::endl;
  initSpecies();
  cout << "5. initializing processes the second time..." << std::endl;
  initProcessSecond();
  cout << "6. constructing lattice..." << std::endl;
  constructLattice();
  cout << "7. setting intersecting compartment list..." << std::endl;
  setIntersectingCompartmentList();
  cout << "8. compartmentalizing lattice..." << std::endl;
  compartmentalizeLattice();
  cout << "9. setting up compartment voxels properties..." << std::endl;
  setCompVoxelProperties();
  resizeProcessLattice();
  cout << "10. initializing processes the third time..." << std::endl;
  initProcessThird();
  cout << "11. printing simulation parameters..." << std::endl;
  updateSpecies();
  storeSimulationParameters();
#ifndef WITH_NO_DUMP
  printSimulationParameters();
#endif
  cout << "12. populating compartments with molecules..." << std::endl;
  populateComps();
  cout << "13. initializing processes the fourth time..." << std::endl;
  initProcessFourth();
  cout << "14. initializing the priority queue..." << std::endl;
  initPriorityQueue();
  cout << "15. initializing processes the fifth time..." << std::endl;
  initProcessFifth();
  cout << "16. initializing processes the last time..." << std::endl;
  initProcessLastOnce();
  cout << "17. finalizing species..." << std::endl;
  finalizeSpecies();
  cout << "18. printing final process parameters..." << std::endl <<
    std::endl;
#ifndef WITH_NO_DUMP
  printProcessParameters();
#endif
  cout << "19. simulation is started..." << std::endl;
  }
}

void SpatiocyteStepper::interrupt(double aTime)
{
  setCurrentTime(aTime); 
  for(std::vector<Process*>::const_iterator 
      i(theExternInterruptedProcesses.begin());
      i != theExternInterruptedProcesses.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->substrateValueChanged(getCurrentTime()); 
    }
  setNextTime(thePriorityQueue.getTop()->getTime());
}

void SpatiocyteStepper::finalizeSpecies()
{
  for(std::vector<Species*>::iterator i(theSpecies.begin());
      i != theSpecies.end(); ++i)
    {
      (*i)->finalizeSpecies();
    }
}

void SpatiocyteStepper::updateSpecies()
{
  for(std::vector<Species*>::iterator i(theSpecies.begin());
      i != theSpecies.end(); ++i)
    {
      (*i)->updateSpecies();
    }
}

unsigned SpatiocyteStepper::getRowSize()
{
  return theRowSize;
}

unsigned SpatiocyteStepper::getLayerSize()
{
  return theLayerSize;
}

unsigned SpatiocyteStepper::getColSize()
{
  return theColSize;
}

unsigned SpatiocyteStepper::getLatticeSize()
{
  return theLattice.size();
}

Point SpatiocyteStepper::getCenterPoint()
{
  return theCenterPoint;
} 

double SpatiocyteStepper::getNormalizedVoxelRadius()
{
  return theNormalizedVoxelRadius;
}

void SpatiocyteStepper::reset(int seed)
{
  gsl_rng_set(getRng(), seed); 
  setCurrentTime(0);
  initProcessSecond();
  clearComps();
  initProcessThird();
  populateComps();
  initProcessFourth();
  initPriorityQueue();
  initProcessFifth();
  finalizeSpecies();
  //checkLattice();
}

Species* SpatiocyteStepper::addSpecies(Variable* aVariable)
{
  std::vector<Species*>::iterator aSpeciesIter(variable2ispecies(aVariable));
  if(aSpeciesIter == theSpecies.end())
    {
      Species *aSpecies(new Species(this, aVariable, theSpecies.size(),
                          (int)aVariable->getValue(), getRng(), VoxelRadius,
                          theLattice));
      theSpecies.push_back(aSpecies);
      return aSpecies;
    }
  return *aSpeciesIter;
}

Species* SpatiocyteStepper::getSpecies(Variable* aVariable)
{
  std::vector<Species*>::iterator aSpeciesIter(variable2ispecies(aVariable));
  if(aSpeciesIter == theSpecies.end())
    {
      return NULL;
    }
  return *aSpeciesIter;
}

std::vector<Species*> SpatiocyteStepper::getSpecies()
{
  return theSpecies;
}

bool SpatiocyteStepper::isBoundaryCoord(unsigned aCoord,
                                        unsigned aDimension)
{
  //This method is only for checking boundaries on a cube:
  unsigned aRow;
  unsigned aLayer;
  unsigned aCol;
  coord2global(aCoord, aRow, aLayer, aCol);
  if(aDimension == 3)
    {
      //If the voxel is on one of the 6 cuboid surfaces:
      if(aRow == 1 || aLayer == 1 || aCol == 1 ||
         aRow == theRowSize-2 || aLayer == theLayerSize-2 || 
         aCol == theColSize-2)
        {
          return true;
        }
    }
  //If it is a surface voxel:
  else
    {
      //If the voxel is on one of the 12 edges of the cube:
      if((aRow <= 1 && aCol <= 1) ||
         (aRow <= 1 && aLayer <= 1) ||
         (aCol <= 1 && aLayer <= 1) ||
         (aRow <= 1 && aCol >= theColSize-2) ||
         (aRow <= 1 && aLayer >= theLayerSize-2) ||
         (aCol <= 1 && aRow >= theRowSize-2) ||
         (aCol <= 1 && aLayer >= theLayerSize-2) ||
         (aLayer <= 1 && aRow >= theRowSize-2) ||
         (aLayer <= 1 && aCol >= theColSize-2) ||
         (aRow >= theRowSize-2 && aCol >= theColSize-2) ||
         (aRow >= theRowSize-2 && aLayer >= theLayerSize-2) ||
         (aCol >= theColSize-2 && aLayer >= theLayerSize-2))
        {
          return true;
        }
    }
  return false;
}

unsigned SpatiocyteStepper::getPeriodicCoord(unsigned aCoord,
                                                 unsigned aDimension,
                                                 Origin* anOrigin)
{
  //This method is only for checking boundaries on a cube:
  unsigned aRow;
  unsigned aLayer;
  unsigned aCol;
  coord2global(aCoord, aRow, aLayer, aCol);
  unsigned nextRow(aRow);
  unsigned nextLayer(aLayer);
  unsigned nextCol(aCol);
  unsigned adj(1);
  if(aDimension == 3)
    {
      adj = 0;
    }
  if(aRow == 1+adj)
    {
      nextRow = theRowSize-(3+adj);
      --anOrigin->row;
    }
  else if(aRow == theRowSize-(2+adj))
    {
      nextRow = 2+adj;
      ++anOrigin->row;
    }
  if(aLayer == 1+adj)
    {
      nextLayer = theLayerSize-(3+adj);
      --anOrigin->layer;
    }
  else if(aLayer == theLayerSize-(2+adj))
    {
      nextLayer = 2+adj;
      ++anOrigin->layer;
    }
  if(aCol == 1+adj)
    {
      nextCol = theColSize-(3+adj);
      --anOrigin->col;
    }
  else if(aCol == theColSize-(2+adj))
    {
      nextCol = 2+adj;
      ++anOrigin->col;
    }
  if(nextRow != aRow || nextCol != aCol || nextLayer != aLayer)
    {
      return nextRow+theRowSize*nextLayer+theRowSize*theLayerSize*nextCol;
    }
  return theNullCoord;
}

Point SpatiocyteStepper::getPeriodicPoint(unsigned aCoord,
                                          unsigned aDimension,
                                          Origin* anOrigin)
{
  unsigned adj(1);
  if(aDimension == 3)
    {
      adj = 0;
    }
  unsigned row(theRowSize-(3+adj)-(1+adj));
  unsigned layer(theLayerSize-(3+adj)-(1+adj));
  unsigned col(theColSize-(3+adj)-(1+adj));
  unsigned aGlobalCol;
  unsigned aGlobalLayer;
  unsigned aGlobalRow;
  coord2global(aCoord, aGlobalRow, aGlobalLayer, aGlobalCol);
  int aRow(aGlobalRow+anOrigin->row*row);
  int aLayer(aGlobalLayer+anOrigin->layer*layer);
  int aCol(aGlobalCol+anOrigin->col*col);
  Point aPoint;
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      aPoint.y = (aCol%2)*theHCPl+theHCPy*aLayer;
      aPoint.z = aRow*2*theNormalizedVoxelRadius+
        ((aLayer+aCol)%2)*theNormalizedVoxelRadius;
      aPoint.x = aCol*theHCPx;
      break;
    case CUBIC_LATTICE:
      aPoint.y = aLayer*2*theNormalizedVoxelRadius;
      aPoint.z = aRow*2*theNormalizedVoxelRadius;
      aPoint.x = aCol*2*theNormalizedVoxelRadius;
      break;
    }
  return aPoint;
}


std::vector<Species*>::iterator
SpatiocyteStepper::variable2ispecies(Variable* aVariable)
{
  for(std::vector<Species*>::iterator i(theSpecies.begin());
      i != theSpecies.end(); ++i)
    {
      if((*i)->getVariable() == aVariable)
        {
          return i;
        }
    }
  return theSpecies.end();
} 

Species* SpatiocyteStepper::variable2species(Variable* aVariable)
{
  for(std::vector<Species*>::iterator i(theSpecies.begin());
      i != theSpecies.end(); ++i)
    {
      if((*i)->getVariable() == aVariable)
        {
          return (*i);
        }
    }
  return NULL;
}

void SpatiocyteStepper::checkModel()
{
  //check if nonHD species are being used by non-SpatiocyteProcesses
  Model::StepperMap aStepperMap(getModel()->getStepperMap());  
  for(Model::StepperMap::const_iterator i(aStepperMap.begin());
      i != aStepperMap.end(); ++i )
    {   
      if(i->second != this)
        {
          std::vector<Process*> aProcessVector(i->second->getProcessVector());
          for(std::vector<Process*>::const_iterator j(aProcessVector.begin());
              j != aProcessVector.end(); ++j)
            {
              Process::VariableReferenceVector aVariableReferenceVector( 
                               (*j)->getVariableReferenceVector());
              for(Process::VariableReferenceVector::const_iterator 
                  k(aVariableReferenceVector.begin());
                  k != aVariableReferenceVector.end(); ++k)
                { 
                  const VariableReference& aNewVariableReference(*k);
                  Variable* aVariable(aNewVariableReference.getVariable()); 
                  for(std::vector<Species*>::iterator m(theSpecies.begin());
                      m !=theSpecies.end(); ++m)
                    {
                      if((*m)->getVariable() == aVariable)
                        {
                          THROW_EXCEPTION(ValueError, 
                            getPropertyInterface().getClassName() +
                            ": " + aVariable->getFullID().asString()  +  
                            " is a non-HD species but it is being used " +
                            "by non-SpatiocyteProcess: " +
                            (*j)->getFullID().asString());
                        }
                    } 
                }
            }
        }
    }
}

void SpatiocyteStepper::checkLattice()
{
  std::vector<int> list;
  for(unsigned i(0); i!=theSpecies.size(); ++i)
    {
      list.push_back(0);
    }
  for(unsigned i(0); i!=theLattice.size(); ++i)
    {
      ++list[theLattice[i].id];
    }
  int volumeCnt(0);
  int surfaceCnt(0);
  for(unsigned i(0); i!=list.size(); ++i)
    {
      std::cout << "i:" << i << " ";
      if(theSpecies[i]->getVariable() != NULL)
        {
          std::cout << theSpecies[i]->getVariable()->getFullID().asString();
          if(theSpecies[i]->getDimension() == 3)
            {
              volumeCnt += list[i];
            }
          else
            {
              surfaceCnt += list[i];
            }
        }
      std::cout << " cnt:" << list[i] << std::endl;
    }
  std::cout << "total volume:" << volumeCnt << std::endl;
  std::cout << "total surface:" << surfaceCnt << std::endl;
  std::cout << "total volume+surface:" << surfaceCnt+volumeCnt << std::endl;
}

void SpatiocyteStepper::initSpecies()
{
  for(std::vector<Species*>::iterator i(theSpecies.begin());
      i != theSpecies.end(); ++i)
    {
      (*i)->initialize(theSpecies.size(), theAdjoiningCoordSize, 
                       theNullCoord, theNullID);
    }
}

void SpatiocyteStepper::broadcastLatticeProperties()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->setLatticeProperties(&theLattice, theAdjoiningCoordSize,
                                     theNullCoord, theNullID);
    }
}

void SpatiocyteStepper::initProcessSecond()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->initializeSecond();
    }
}

void SpatiocyteStepper::printProcessParameters()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->printParameters();
    }
  std::cout << std::endl;
}

void SpatiocyteStepper::resizeProcessLattice()
{
  unsigned startCoord(theLattice.size());
  unsigned endCoord(startCoord);
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      endCoord += aProcess->getLatticeResizeCoord(endCoord);
    }
  //Save the coords of molecules before resizing theLattice
  //because the voxel address pointed by molecule list will become
  //invalid once it is resized:
  for(unsigned i(0); i != theSpecies.size(); ++i)
    {
      theSpecies[i]->saveCoords();
    }
  theLattice.resize(endCoord);
  for(unsigned i(startCoord); i != endCoord; ++i)
    {
      theLattice[i].coord = i;
    }
  //Update the molecule list with the new voxel addresses:
  for(unsigned i(0); i != theSpecies.size(); ++i)
    {
      theSpecies[i]->updateMoleculePointers();
    }
}

void SpatiocyteStepper::initProcessThird()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->initializeThird();
    }
}

void SpatiocyteStepper::initProcessFourth()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->initializeFourth();
    }
}

void SpatiocyteStepper::initProcessFifth()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->initializeFifth();
    }
  setStepInterval(thePriorityQueue.getTop()->getTime()-getCurrentTime());
}

void SpatiocyteStepper::initProcessLastOnce()
{
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      SpatiocyteProcessInterface*
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      aProcess->initializeLastOnce();
    }
}

void SpatiocyteStepper::initPriorityQueue()
{
  const double aCurrentTime(getCurrentTime());
  thePriorityQueue.clear();
  for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
      i != theProcessVector.end(); ++i)
    {      
      Process* const aProcess(*i);
      SpatiocyteProcessInterface*
        aSpatiocyteProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      if(aSpatiocyteProcess)
        {
          aSpatiocyteProcess->setTime(aCurrentTime+aProcess->getStepInterval());
          aSpatiocyteProcess->setPriorityQueue(&thePriorityQueue);
          //Not all SpatiocyteProcesses are inserted into the priority queue.
          //Only the following processes are inserted in the PriorityQueue and
          //executed at simulation steps according to their execution times:
          if(aSpatiocyteProcess->getIsPriorityQueued())
            {
              aSpatiocyteProcess->setQueueID(
                                   thePriorityQueue.push(aSpatiocyteProcess));
              //ExternInterrupted processes are processes which are
              //interrupted by non SpatiocyteStepper processes such as
              //ODE processes:
              if(aSpatiocyteProcess->getIsExternInterrupted())
                {
                  theExternInterruptedProcesses.push_back(aProcess);
                }
            }
        }
      //Interruption between SpatiocyteProcesses:
      //Only ReactionProcesses can interrupt other processes because only 
      //they can change the number of molecules. 
      //This method is called to set the list of processes which will be
      //interrupted by the current ReactionProcess:
      ReactionProcessInterface*
        aReactionProcess(dynamic_cast<ReactionProcessInterface*>(*i));
      if(aReactionProcess)
        {
          aReactionProcess->setInterruption(theProcessVector);
        }
    }
}

void SpatiocyteStepper::populateComps()
{
  for(std::vector<Comp*>::const_iterator i(theComps.begin());
      i != theComps.end(); ++i)
    {
      populateComp(*i);
    }
}

void SpatiocyteStepper::clearComps()
{
  for(std::vector<Comp*>::const_iterator i(theComps.begin());
      i != theComps.end(); ++i)
    {
      clearComp(*i);
    }
}


inline void SpatiocyteStepper::step()
{
  do
    {
      thePriorityQueue.getTop()->fire();
    }
  while(thePriorityQueue.getTop()->getTime() == getCurrentTime());
  setNextTime(thePriorityQueue.getTop()->getTime());
  //checkLattice();
} 


void SpatiocyteStepper::registerComps()
{
  System* aRootSystem(getModel()->getRootSystem());
  std::vector<Comp*> allSubs;
  //The root Comp is theComps[0]
  theComps.push_back(registerComp(aRootSystem, &allSubs));
  //After this we will create an species to get an ID to represent
  //NULL Comps. So let us specify the correct
  //size of the biochemical species to be simulated before additional
  //non-biochemical species are created:
  theBioSpeciesSize = theSpecies.size();
  //Create one last species to represent a NULL Comp. This is for
  //voxels that do not belong to any Comps:
  Species* aSpecies(new Species(this, NULL, theSpecies.size(), 0, getRng(),
                                VoxelRadius, theLattice));
  theSpecies.push_back(aSpecies);
  aSpecies->setComp(NULL);
  theNullID = aSpecies->getID(); 
  //Expand the tree of immediate subComps into single list such that
  //the super Comps come first while the subComps 
  //come later in the list:
  std::vector<Comp*> Comps(theComps[0]->immediateSubs);
  while(!Comps.empty())
    {
      std::vector<Comp*> subComps;
      for(unsigned i(0); i != Comps.size(); ++i)
        {
          theComps.push_back(Comps[i]);
          for(unsigned j(0);
              j != Comps[i]->immediateSubs.size(); ++j)
            {
              subComps.push_back(Comps[i]->immediateSubs[j]);
            }
        }
      Comps = subComps;
    }
}

//allSubs contains all the subComps (child, grand child, great grand
//child, etc). Used to calculate the total number of Comp voxels.
Comp* SpatiocyteStepper::registerComp(System* aSystem,
                                            std::vector<Comp*>* allSubs)
{ 
  //We execute this function to register the System, and its subsystems
  //recursively.
  Comp* aComp(new Comp);
  aComp->minRow = UINT_MAX;
  aComp->minCol = UINT_MAX;
  aComp->minLayer = UINT_MAX;
  aComp->maxRow = 0;
  aComp->maxCol = 0;
  aComp->maxLayer = 0;
  aComp->lengthX = 0;
  aComp->lengthY = 0;
  aComp->lengthZ = 0;
  aComp->originX = 0;
  aComp->originY = 0;
  aComp->originZ = 0;
  aComp->rotateX = 0;
  aComp->rotateY = 0;
  aComp->rotateZ = 0;
  aComp->xyPlane = 0;
  aComp->xzPlane = 0;
  aComp->yzPlane = 0;
  aComp->specVolume = 0;
  aComp->system = aSystem;
  aComp->surfaceSub = NULL;
  aComp->diffusiveComp = NULL;
  //Default Comp geometry is Cuboid:
  aComp->geometry = 0;
  //Default is volume Comp:
  aComp->dimension = 3;
  if(getVariable(aSystem, "DIMENSION"))
    { 
      const double aDimension(aSystem->getVariable("DIMENSION")->getValue());
      aComp->dimension = aDimension;
    }
  if(aComp->dimension == 3)
    {
      if(getVariable(aSystem, "GEOMETRY"))
        { 
          aComp->geometry = aSystem->getVariable("GEOMETRY")->getValue();
        }
      if(getVariable(aSystem, "LENGTHX"))
        {
          aComp->lengthX = aSystem->getVariable("LENGTHX")->getValue();
        }
      if(getVariable(aSystem, "LENGTHY"))
        {
          aComp->lengthY = aSystem->getVariable("LENGTHY")->getValue();
        }
      if(getVariable(aSystem, "LENGTHZ"))
        {
          aComp->lengthZ = aSystem->getVariable("LENGTHZ")->getValue();
        }
      if(getVariable(aSystem, "ORIGINX"))
        {
          aComp->originX = aSystem->getVariable("ORIGINX")->getValue();
        }
      if(getVariable(aSystem, "ORIGINY"))
        {
          aComp->originY = aSystem->getVariable("ORIGINY")->getValue();
        }
      if(getVariable(aSystem, "ORIGINZ"))
        {
          aComp->originZ = aSystem->getVariable("ORIGINZ")->getValue();
        }
      if(getVariable(aSystem, "ROTATEX"))
        {
          aComp->rotateX = aSystem->getVariable("ROTATEX")->getValue();
        }
      if(getVariable(aSystem, "ROTATEY"))
        {
          aComp->rotateY = aSystem->getVariable("ROTATEY")->getValue();
        }
      if(getVariable(aSystem, "ROTATEZ"))
        {
          aComp->rotateZ = aSystem->getVariable("ROTATEZ")->getValue();
        }
      if(getVariable(aSystem, "XYPLANE"))
        {
          aComp->xyPlane = aSystem->getVariable("XYPLANE")->getValue();
        }
      if(getVariable(aSystem, "XZPLANE"))
        {
          aComp->xzPlane = aSystem->getVariable("XZPLANE")->getValue();
        }
      if(getVariable(aSystem, "YZPLANE"))
        {
          aComp->yzPlane = aSystem->getVariable("YZPLANE")->getValue();
        }
    }
  registerCompSpecies(aComp);
  //Systems contains all the subsystems of a System.
  //For example /membrane is the subsystem of /:
  FOR_ALL(System::Systems, aSystem->getSystems())
    {
      std::vector<Comp*> currSubs; 
      Comp* aSubComp(registerComp(i->second, &currSubs)); 
      allSubs->push_back(aSubComp);
      for(unsigned j(0); j != currSubs.size(); ++j)
        {
          allSubs->push_back(currSubs[j]);
        }
      aComp->immediateSubs.push_back(aSubComp);
      if(aSubComp->dimension == 2)
      {
          aSubComp->geometry = aComp->geometry;
          aSubComp->specVolume = aComp->specVolume;
          aSubComp->lengthX = aComp->lengthX;
          aSubComp->lengthY = aComp->lengthY;
          aSubComp->lengthZ = aComp->lengthZ;
          aSubComp->originX = aComp->originX;
          aSubComp->originY = aComp->originY;
          aSubComp->originZ = aComp->originZ;
          aSubComp->xyPlane = aComp->xyPlane;
          aSubComp->xzPlane = aComp->xzPlane;
          aSubComp->yzPlane = aComp->yzPlane;
          aSubComp->rotateX = aComp->rotateX;
          aSubComp->rotateY = aComp->rotateY;
          aSubComp->rotateZ = aComp->rotateZ;

          for(unsigned i(0); i != aSubComp->lineSubs.size(); ++i)
          {
              Comp* lineComp(aSubComp->lineSubs[i]);
              lineComp->geometry = aComp->geometry;
              lineComp->specVolume = aComp->specVolume;
              lineComp->lengthX = aComp->lengthX;
              lineComp->lengthY = aComp->lengthY;
              lineComp->lengthZ = aComp->lengthZ;
              lineComp->originX = aComp->originX;
              lineComp->originY = aComp->originY;
              lineComp->originZ = aComp->originZ;
              lineComp->xyPlane = aComp->xyPlane;
              lineComp->xzPlane = aComp->xzPlane;
              lineComp->yzPlane = aComp->yzPlane;
              lineComp->rotateX = aComp->rotateX;
              lineComp->rotateY = aComp->rotateY;
              lineComp->rotateZ = aComp->rotateZ;
          }

          aComp->surfaceSub = aSubComp;
      }
      else if(aSubComp->dimension == 1)
      {
          // Properties of aComp (dimension == 2) is not fully set here yet.
          aComp->lineSubs.push_back(aSubComp);
      }
    }
  aComp->allSubs = *allSubs;
  return aComp;
}

void SpatiocyteStepper::setCompsProperties()
{
  for(unsigned i(0); i != theComps.size(); ++i)
    {
      setCompProperties(theComps[i]);
    }
}

void SpatiocyteStepper::setCompsCenterPoint()
{
  for(unsigned i(0); i != theComps.size(); ++i)
    {
      setCompCenterPoint(theComps[i]);
    }
}

void SpatiocyteStepper::registerCompSpecies(Comp* aComp)
{
  System* aSystem(aComp->system);
  FOR_ALL(System::Variables, aSystem->getVariables())
    {
      Variable* aVariable(i->second);
      if(aVariable->getID() == "VACANT")
        {
          aComp->enclosed = aVariable->getValue();
          //Set the number of vacant molecules to be always 0 because
          //when we populate lattice we shouldn't create more vacant
          //molecules than the ones already created for the Comp:
          aVariable->setValue(0);
          Species* aSpecies(addSpecies(aVariable));
          aComp->vacantSpecies = aSpecies;
          aSpecies->setVacantSpecies(aSpecies);
          aComp->vacantID = aSpecies->getID(); //remove this
          aSpecies->setIsCompVacant();
        }
      std::vector<Species*>::iterator j(variable2ispecies(aVariable));
      if(j != theSpecies.end())
        {
          aComp->species.push_back(*j);
          (*j)->setComp(aComp);
          (*j)->setDimension(aComp->dimension);
        }
    }
}

void SpatiocyteStepper::setLatticeProperties()
{
  Comp* aRootComp(theComps[0]);
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      theAdjoiningCoordSize = 12;
      theHCPl = theNormalizedVoxelRadius/sqrt(3); 
      theHCPx = theNormalizedVoxelRadius*sqrt(8.0/3); //Lx
      theHCPy = theNormalizedVoxelRadius*sqrt(3); //Ly
      break;
    case CUBIC_LATTICE:
      theAdjoiningCoordSize = 6;
      break;
    }
  if(aRootComp->geometry == CUBOID)
    {
      //We do not give any leeway space between the simulation boundary
      //and the cell boundary if it is CUBOID to support
      //periodic boundary conditions:
      theCenterPoint.z = aRootComp->lengthZ/2; //row
      theCenterPoint.y = aRootComp->lengthY/2; //layer
      theCenterPoint.x = aRootComp->lengthX/2; //column
    }
  else
    {
      switch(LatticeType)
        {
        case HCP_LATTICE: 
          theCenterPoint.z = aRootComp->lengthZ/2+4*
            theNormalizedVoxelRadius; //row
          theCenterPoint.y = aRootComp->lengthY/2+2*theHCPy; //layer
          theCenterPoint.x = aRootComp->lengthX/2+2*theHCPx; //column
          break;
        case CUBIC_LATTICE:
          theCenterPoint.z = aRootComp->lengthZ/2+8*theNormalizedVoxelRadius;
          theCenterPoint.y = aRootComp->lengthY/2+8*theNormalizedVoxelRadius;
          theCenterPoint.x = aRootComp->lengthX/2+8*theNormalizedVoxelRadius;
          break;
        }
    }
  aRootComp->centerPoint = theCenterPoint; 
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      theRowSize = (unsigned)rint((theCenterPoint.z)/
                                      (theNormalizedVoxelRadius));
      theLayerSize = (unsigned)rint((theCenterPoint.y*2)/theHCPy);
      theColSize = (unsigned)rint((theCenterPoint.x*2)/theHCPx);
      break;
    case CUBIC_LATTICE:
      theRowSize = (unsigned)rint((theCenterPoint.z)/
                                      (theNormalizedVoxelRadius));
      theLayerSize = (unsigned)rint((theCenterPoint.y)/
                                      (theNormalizedVoxelRadius));
      theColSize = (unsigned)rint((theCenterPoint.x)/
                                      (theNormalizedVoxelRadius));
      break;
    }
  //For the CUBOID cell geometry, we need to readjust the size of
  //row, layer and column according to the boundary condition of its surfaces
  //to reflect the correct volume. This is because periodic boundary will
  //consume a layer of the surface voxels:
  if(aRootComp->geometry == CUBOID)
    {
      //We need to increase the row, layer and col size by 2 because
      //the entire volume must be surrounded by nullID voxels to avoid
      //self-homodimerization reaction.
      theRowSize += 2;
      theColSize += 2;
      theLayerSize += 2;
      readjustSurfaceBoundarySizes(); 
    }

  //You should only resize theLattice once here. You should never
  //push or readjust the capacity of theLattice since all the pointers
  //to the voxel will become invalid once you do that:
  //Also add one more voxel for the nullVoxel:
  theLattice.resize(theRowSize*theLayerSize*theColSize+1);
  //Initialize the null coord:
  theNullCoord = theRowSize*theLayerSize*theColSize;
  theLattice[theNullCoord].id = theNullID;
}

void SpatiocyteStepper::storeSimulationParameters()
{
  for(unsigned i(0); i != theComps.size(); ++i)
    {
      Comp* aComp(theComps[i]); 
      if(aComp->dimension == 2)
        {
          aComp->actualArea =  (72*pow(VoxelRadius,2))*
            aComp->vacantSpecies->size()/(6*pow(2,0.5)+4*pow(3,0.5)+
                                         3*pow(6, 0.5));
          setSystemSize(aComp->system, aComp->actualArea*1e+2);
        }
      else // (aComp->dimension == 3)
        { 
          int voxelCnt(aComp->vacantSpecies->size());
          for(unsigned j(0); j != aComp->allSubs.size(); ++j)
            {
              voxelCnt += aComp->allSubs[j]->vacantSpecies->size();
            }
          aComp->actualVolume = (4*pow(2,0.5)*pow(VoxelRadius,3))*
            voxelCnt;
          setSystemSize(aComp->system, aComp->actualVolume*1e+3);
        }
    }
}

void SpatiocyteStepper::setSystemSize(System* aSystem, double aSize)
{
  Variable* aVariable(getVariable(aSystem, "SIZE"));
  if(aVariable)
    {
      aVariable->setValue(aSize);
    }
  else
    {
      createVariable(aSystem, "SIZE")->setValue(aSize);
    }
}

Variable* SpatiocyteStepper::createVariable(System* aSystem, String anID)
{
  String anEntityType("Variable");
  SystemPath aSystemPath(aSystem->getSystemPath());
  aSystemPath.push_back(aSystem->getID());
  FullID aFullID(anEntityType, aSystemPath, anID);
  Variable* aVariable(reinterpret_cast<Variable*>(
                              getModel()->createEntity("Variable", aFullID)));
  aVariable->setValue(0);
  return aVariable;
}

Species* SpatiocyteStepper::createSpecies(System* aSystem, String anID)
{
  Variable* aVariable(createVariable(aSystem, anID));
  return addSpecies(aVariable);
}

void SpatiocyteStepper::printSimulationParameters()
{
  std::cout << std::endl;
  std::cout << "   Voxel radius, r_v:" << VoxelRadius << " m" << std::endl;
  std::cout << "   Simulation height:" << theCenterPoint.y*2*VoxelRadius*2 <<
    " m" << std::endl;
  std::cout << "   Simulation width:" << theCenterPoint.z*2*VoxelRadius*2 << 
    " m" << std::endl;
  std::cout << "   Simulation length:" << theCenterPoint.x*2*VoxelRadius*2 <<
    " m" << std::endl;
  std::cout << "   Layer size:" << theLayerSize << std::endl;
  std::cout << "   Row size:" << theRowSize << std::endl;
  std::cout << "   Column size:" << theColSize << std::endl;
  std::cout << "   Total allocated voxels:" << 
    theRowSize*theLayerSize*theColSize << std::endl;
  for(unsigned i(0); i != theComps.size(); ++i)
    {
      Comp* aComp(theComps[i]);
      double aSpecVolume(aComp->specVolume);
      double aSpecArea(aComp->specArea);
      double anActualVolume(aComp->actualVolume);
      double anActualArea(aComp->actualArea);
      switch(aComp->geometry)
        {
        case CUBOID:
          std::cout << "   Cuboid ";
          break;
        case ELLIPSOID:
          std::cout << "   Ellipsoid ";
          break;
        case CYLINDER:
          std::cout << "   Cylinder (radius=" << aComp->lengthY*VoxelRadius
            << "m, length=" << (aComp->lengthX)*VoxelRadius*2 << "m) ";
          break;
        case ROD:
          std::cout << "   Rod (radius=" << aComp->lengthY*VoxelRadius << 
            "m, cylinder length=" <<
            (aComp->lengthX-aComp->lengthY*2)*VoxelRadius*2 << "m) ";
          break;
        }
      std::cout << aComp->system->getFullID().asString();
      switch (aComp->dimension)
      { 
      case 1:
          std::cout << " Line compartment:" << std::endl;
          break;
      case 2:
          std::cout << " Surface compartment:" << std::endl;
          std::cout << "     [" << int(aSpecArea*(6*sqrt(2)+4*sqrt(3)+3*sqrt(6))/
                              (72*VoxelRadius*VoxelRadius)) << 
            "] Specified surface voxels {n_s = S_specified*"
            << "(6*2^0.5+4*3^0.5+3*6^0.5)/(72*r_v^2}" << std::endl;
          std::cout << "     [" << aComp->vacantSpecies->size() <<
            "] Actual surface voxels {n_s}" << std::endl;
          std::cout << "     [" << aSpecArea << " m^2] Specified surface area " <<
            "{S_specified}" << std::endl;
          std::cout << "     [" << anActualArea << " m^2] Actual surface area " <<
            "{S = (72*r_v^2)*n_s/(6*2^0.5+4*3^0.5+3*6^0.5)}" << std::endl;
          break;
      case 3:
      default:
          std::cout << " Volume compartment:" << std::endl;
          int voxelCnt(aComp->vacantSpecies->size());
          for(unsigned j(0); j != aComp->allSubs.size(); ++j)
            {
              //Don't include the comp surface voxels when you count the
              //total volume voxels:
              if(aComp->surfaceSub != aComp->allSubs[j])
                {
                  voxelCnt += aComp->allSubs[j]->vacantSpecies->size();
                }
            }
          std::cout << "     [" << int(aSpecVolume/(4*sqrt(2)*pow(VoxelRadius, 3))) << 
            "] Specified volume voxels {n_v = V_specified/(4*2^0.5*r_v^3)}" <<
          std::endl;  
          std::cout << "     [" << voxelCnt << "] Actual volume voxels {n_v}"  << std::endl;
          std::cout << "     [" << aSpecVolume << " m^3] Specified volume {V_specified}"
            << std::endl; 
          std::cout << "     [" << anActualVolume << " m^3] Actual volume " <<
            "{V = (4*2^0.5*r_v^3)*n_v}" << std::endl; 
      }
    }
  std::cout << std::endl;
}

void SpatiocyteStepper::readjustSurfaceBoundarySizes()
{
  Comp* aRootComp(theComps[0]);
  //[XY, XZ, YZ]PLANE: the boundary type of the surface when 
  //the geometry of the root Comp is CUBOID.
  //Where the root cuboid compartment is enclosed with a surface compartment,
  //the boundary type can be either REFLECTIVE, REMOVE_UPPER, REMOVE_LOWER or
  //REMOVE_BOTH. To make the actualVolume of the root compartment equivalent
  //to the specVolume we need to increase the size of the row, col and layer
  //according to the additional voxels required to occupy the surface voxels.
  if(aRootComp->surfaceSub)
    {
      if(aRootComp->xyPlane == REFLECTIVE)
        {
          theRowSize += 2;
        }
      else if(aRootComp->xyPlane == REMOVE_UPPER ||
              aRootComp->xyPlane == REMOVE_LOWER)
        {
          theRowSize += 1;
        }
      if(aRootComp->yzPlane == REFLECTIVE)
        {
          theColSize += 2;
        }
      else if(aRootComp->yzPlane == REMOVE_UPPER ||
              aRootComp->yzPlane == REMOVE_LOWER)
        {
          theColSize += 1;
        }
      if(aRootComp->xzPlane == REFLECTIVE)
        {
          theLayerSize += 2;
        }
      else if(aRootComp->xzPlane == REMOVE_UPPER ||
              aRootComp->xzPlane == REMOVE_LOWER)
        {
          theLayerSize += 1;
        }
    }
  else
    {
      //Boundary type can also be either PERIODIC or REFLECTIVE when there is
      //no surface compartment for the root compartment.
      //Increase the size of [row,layer,col] by one voxel and make them odd
      //sized if the system uses periodic boundary conditions.
      if(aRootComp->yzPlane == PERIODIC)
        { 
          if(theColSize%2 != 1)
            {
              theColSize += 1;
            }
          else
            {
              theColSize += 2;
            }
        }
      if(aRootComp->xzPlane == PERIODIC)
        {
          if(theLayerSize%2 != 1)
            {
              theLayerSize +=1;
            }
          else
            {
              theLayerSize += 2;
            }
        }
      if(aRootComp->xyPlane == PERIODIC)
        {
          if(theRowSize%2 != 1)
            {
              theRowSize += 1;
            }
          else
            {
              theRowSize += 2;
            }
        }
    }
  if(isPeriodicEdge)
    {
      if(theColSize%2 == 1)
        {
          theColSize += 1;
        }
      if(theLayerSize%2 == 1)
        {
          theLayerSize +=1;
        }
      if(theRowSize%2 == 1)
        {
          theRowSize += 1;
        }
    }
}

void SpatiocyteStepper::constructLattice()
{ 
  Comp* aRootComp(theComps[0]);
  unsigned aSize(theRowSize*theLayerSize*theColSize);
  unsigned a(0);
  unsigned short rootID(aRootComp->vacantSpecies->getID());
  for(std::vector<Voxel>::iterator i(theLattice.begin()); a != aSize; ++i, ++a)
    { 
      (*i).coord = a;
      (*i).adjoiningCoords = new unsigned[theAdjoiningCoordSize];
      (*i).initAdjoins = NULL;
      (*i).diffuseSize = theAdjoiningCoordSize;
      (*i).adjoiningSize = theAdjoiningCoordSize;
      (*i).point = NULL;
      unsigned aCol(a/(theRowSize*theLayerSize)); 
      unsigned aLayer((a%(theRowSize*theLayerSize))/theRowSize); 
      unsigned aRow((a%(theRowSize*theLayerSize))%theRowSize); 
      if(aRootComp->geometry == CUBOID || isInsideCoord(a, aRootComp, 0))
        {
          //By default, the voxel is vacant and we set it to the root id:
          (*i).id = rootID;
          for(unsigned j(0); j != theAdjoiningCoordSize; ++j)
            { 
              // By default let the adjoining voxel pointer point to the 
              // source voxel (i.e., itself)
              (*i).adjoiningCoords[j] = a;
            } 
          concatenateVoxel(*i, aRow, aLayer, aCol);
        }
      else
        {
          //We set id = theNullID if it is an invalid voxel, i.e., no molecules
          //will occupy it:
          (*i).id = theNullID;
          //Concatenate some of the null voxels close to the surface:
          if(isInsideCoord(a, aRootComp, 4))
            {
              concatenateVoxel(*i, aRow, aLayer, aCol);
            }
        }
    }
  if(aRootComp->geometry == CUBOID)
    {
      concatenatePeriodicSurfaces();
    }
}

void SpatiocyteStepper::setPeriodicEdge()
{
  isPeriodicEdge = true;
}

bool SpatiocyteStepper::isPeriodicEdgeCoord(unsigned aCoord, Comp* aComp)
{
  unsigned aRow;
  unsigned aLayer;
  unsigned aCol;
  coord2global(aCoord, aRow, aLayer, aCol);
  if(aComp->system->getSuperSystem()->isRootSystem() &&
     ((aRow <= 1 && aCol <= 1) ||
      (aRow <= 1 && aLayer <= 1) ||
      (aCol <= 1 && aLayer <= 1) ||
      (aRow <= 1 && aCol >= theColSize-2) ||
      (aRow <= 1 && aLayer >= theLayerSize-2) ||
      (aCol <= 1 && aRow >= theRowSize-2) ||
      (aCol <= 1 && aLayer >= theLayerSize-2) ||
      (aLayer <= 1 && aRow >= theRowSize-2) ||
      (aLayer <= 1 && aCol >= theColSize-2) ||
      (aRow >= theRowSize-2 && aCol >= theColSize-2) ||
      (aRow >= theRowSize-2 && aLayer >= theLayerSize-2) ||
      (aCol >= theColSize-2 && aLayer >= theLayerSize-2)))
    {
      return true;
    }
  return false;
}

bool SpatiocyteStepper::isRemovableEdgeCoord(unsigned aCoord, Comp* aComp)
{
  unsigned aRow;
  unsigned aLayer;
  unsigned aCol;
  coord2global(aCoord, aRow, aLayer, aCol);
  int sharedCnt(0);
  int removeCnt(0);
  //Minus 1 to maxRow to account for surfaces that use two rows to envelope the
  //volume:
  if(aRow >= aComp->maxRow-1)
    {
      ++sharedCnt;
      if(aComp->xyPlane == REMOVE_UPPER || aComp->xyPlane == REMOVE_BOTH)
        {
          ++removeCnt;
        }
    }
  //Add 1 to minRow to account for surfaces that use two rows to envelope the
  //volume:
  if(aRow <= aComp->minRow+1)
    {
      ++sharedCnt;
      if(aComp->xyPlane == REMOVE_LOWER || aComp->xyPlane == REMOVE_BOTH)
        {
          ++removeCnt;
        }
    }
  if(aLayer >= aComp->maxLayer-1)
    {
      ++sharedCnt;
      if(aComp->xzPlane == REMOVE_UPPER || aComp->xzPlane == REMOVE_BOTH)
        {
          ++removeCnt;
        }
    }
  if(aLayer <= aComp->minLayer+1)
    {
      ++sharedCnt;
      if(aComp->xzPlane == REMOVE_LOWER || aComp->xzPlane == REMOVE_BOTH)
        {
          ++removeCnt;
        }
    }
  if(aCol >= aComp->maxCol)
    {
      ++sharedCnt;
      if(aComp->yzPlane == REMOVE_UPPER || aComp->yzPlane == REMOVE_BOTH)
        {
          ++removeCnt;
        }
    }
  if(aCol <= aComp->minCol) 
    {
      ++sharedCnt;
      if(aComp->yzPlane == REMOVE_LOWER || aComp->yzPlane == REMOVE_BOTH)
        {
          ++removeCnt;
        }
    }
  if(!removeCnt)
    {
      return false;
    }
  else
    {
      return sharedCnt == removeCnt;
    }
}

void SpatiocyteStepper::concatenateVoxel(Voxel& aVoxel,
                                         unsigned aRow,
                                         unsigned aLayer,
                                         unsigned aCol)
{
  unsigned aCoord(aRow+
                      theRowSize*aLayer+
                      theRowSize*theLayerSize*aCol);
  if(aRow > 0)
    { 
      concatenateRows(aVoxel, aCoord, aRow-1, aLayer, aCol); 
    } 
  if(aLayer > 0)
    {
      concatenateLayers(aVoxel, aCoord, aRow, aLayer-1, aCol); 
    }
  if(aCol > 0)
    { 
      concatenateCols(aVoxel, aCoord, aRow, aLayer, aCol-1); 
    }
}

Variable* SpatiocyteStepper::getVariable(System* aSystem, String const& anID)
{
  FOR_ALL(System::Variables, aSystem->getVariables())
    {
      Variable* aVariable(i->second);
      if(aVariable->getID() == anID)
        {
          return aVariable;
        }
    }
  return NULL;
}

void SpatiocyteStepper::setCompProperties(Comp* aComp)
{
  System* aSystem(aComp->system);
  double aRadius(0);
  switch(aComp->geometry)
    {
    case CUBOID:
      if(!aComp->lengthX)
        {
          THROW_EXCEPTION(NotFound,
                          "Property LENGTHX of the Cuboid Comp " +
                          aSystem->getFullID().asString() + " is not defined.");
        }
      if(!aComp->lengthY)
        {
          THROW_EXCEPTION(NotFound,
                          "Property LENGTHY of the Cuboid Comp " +
                          aSystem->getFullID().asString() + " is not defined.");
        }
      if(!aComp->lengthZ)
        {
          THROW_EXCEPTION(NotFound,
                          "Property LENGTHZ of the Cuboid Comp " +
                          aSystem->getFullID().asString() + " is not defined.");
        }
      aComp->specVolume = aComp->lengthX*aComp->lengthY*
        aComp->lengthZ;
      aComp->specArea = getCuboidSpecArea(aComp);
      break;
    case ELLIPSOID:
      if(!aComp->lengthX)
        {
          THROW_EXCEPTION(NotFound,
                          "Property LENGTHX of the Ellipsoid Comp " +
                          aSystem->getFullID().asString() + " is not defined.");
        }
      if(!aComp->lengthY)
        {
          THROW_EXCEPTION(NotFound,
                          "Property LENGTHY of the Ellipsoid Comp " +
                          aSystem->getFullID().asString() + " is not defined.");
        }
      if(!aComp->lengthZ)
        {
          THROW_EXCEPTION(NotFound,
                          "Property LENGTHZ of the Ellipsoid Comp " +
                          aSystem->getFullID().asString() + " is not defined.");
        }
      aComp->specVolume = 4*M_PI*aComp->lengthX*
        aComp->lengthY* aComp->lengthZ/24;
      aComp->specArea = 4*M_PI*
        pow((pow(aComp->lengthX/2, 1.6075)*
             pow(aComp->lengthY/2, 1.6075)+
             pow(aComp->lengthX/2, 1.6075)*
             pow(aComp->lengthZ/2, 1.6075)+
             pow(aComp->lengthY/2, 1.6075)*
             pow(aComp->lengthZ/2, 1.6075))/ 3, 1/1.6075); 
      break;
    case CYLINDER:
      if(!aComp->lengthX)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHX of the Cylinder Comp "
                          + aSystem->getFullID().asString() + " not defined." );
        }
      if(!aComp->lengthY)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHY of the Cylinder Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies its diameter, not defined." );
        }
      aRadius = aComp->lengthY/2;
      aComp->specVolume = (aComp->lengthX)*(M_PI*aRadius*aRadius);
      aComp->lengthZ = aComp->lengthY;
      aComp->specArea = 2*M_PI*aRadius*(aComp->lengthX);
      break;
    case ROD:
      if(!aComp->lengthX)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHX of the Rod Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the rod length (cylinder length + "
                          + "2*hemisphere radius), not defined." );
        }
      if(!aComp->lengthY)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHY of the Rod Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies its diameter, not defined." );
        }
      aRadius = aComp->lengthY/2;
      aComp->specVolume = (aComp->lengthX+(4*aRadius/3)-(2*aRadius))*
        (M_PI*aRadius*aRadius);
      aComp->lengthZ = aComp->lengthY;
      aComp->specArea = 4*M_PI*aRadius*aRadius+
        2*M_PI*aRadius*(aComp->lengthX-2*aRadius);
      break;
    case PYRAMID:
      if(!aComp->lengthX)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHX of the Pyramid Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the length of the pyramid, "
                          + "not defined." );
        }
      if(!aComp->lengthY)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHY of the Pyramid Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the height of the pyramid, "
                          + "not defined." );
        }
      if(!aComp->lengthZ)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHZ of the Pyramid Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the depth of the pyramid, "
                          + "not defined." );
        }
      aComp->specVolume = aComp->lengthX*aComp->lengthY*aComp->lengthZ/3;
      aRadius = sqrt(pow(aComp->lengthX/2,2)+pow(aComp->lengthY,2));
      aComp->specArea =  aComp->lengthZ*aComp->lengthX+
        2*(aComp->lengthZ*aRadius)+2*(aComp->lengthX*aRadius);
      break;
    case ERYTHROCYTE:
      if(!aComp->lengthX)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHX of the Erythrocyte Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the length of the Erythrocyte, "
                          + "not defined." );
        }
      if(!aComp->lengthY)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHY of the Erythrocyte Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the height of the Erythrocyte, "
                          + "not defined." );
        }
      if(!aComp->lengthZ)
        {
          THROW_EXCEPTION(NotFound, "Property LENGTHZ of the Erythrocyte Comp "
                          + aSystem->getFullID().asString() + ", which "
                          + "specifies the depth of the Erythrocyte, "
                          + "not defined." );
        }
      aComp->specVolume = 4*M_PI*aComp->lengthX*
        aComp->lengthY* aComp->lengthZ/24;
      aComp->specArea = 4*M_PI*
        pow((pow(aComp->lengthX/2, 1.6075)*
             pow(aComp->lengthY/2, 1.6075)+
             pow(aComp->lengthX/2, 1.6075)*
             pow(aComp->lengthZ/2, 1.6075)+
             pow(aComp->lengthY/2, 1.6075)*
             pow(aComp->lengthZ/2, 1.6075))/ 3, 1/1.6075); 
      break;
    }
  aComp->lengthX /= VoxelRadius*2;
  aComp->lengthY /= VoxelRadius*2;
  aComp->lengthZ /= VoxelRadius*2;
}

void SpatiocyteStepper::setCompCenterPoint(Comp* aComp)
{
  System* aSystem(aComp->system);
  System* aSuperSystem(aSystem->getSuperSystem());
  if(aComp->dimension == 2)
    {
      aSystem = aComp->system->getSuperSystem();
      aSuperSystem = aSystem->getSuperSystem();
    }
  else if(aComp->dimension == 1)
    {
      aSystem = aComp->system->getSuperSystem()->getSuperSystem();
      aSuperSystem = aSystem->getSuperSystem();
    }
  Comp* aSuperComp(system2Comp(aSuperSystem));
  //The center with reference to the immediate super system:
  aComp->centerPoint = aSuperComp->centerPoint;
  aComp->centerPoint.x += aComp->originX*aSuperComp->lengthX/2;
  aComp->centerPoint.y += aComp->originY*aSuperComp->lengthY/2;
  aComp->centerPoint.z += aComp->originZ*aSuperComp->lengthZ/2;
}

double SpatiocyteStepper::getCuboidSpecArea(Comp* aComp)
{
  double anArea(0);
  if(aComp->xyPlane == UNIPERIODIC || 
     aComp->xyPlane == REMOVE_UPPER ||
     aComp->xyPlane == REMOVE_LOWER)
    { 
      anArea += aComp->lengthX*aComp->lengthY; 
    }
  else if(aComp->xyPlane == REFLECTIVE)
    {
      anArea += 2*aComp->lengthX*aComp->lengthY; 
    }
  if(aComp->xzPlane == UNIPERIODIC || 
     aComp->xzPlane == REMOVE_UPPER ||
     aComp->xzPlane == REMOVE_LOWER)
    { 
      anArea += aComp->lengthX*aComp->lengthZ; 
    }
  else if(aComp->xzPlane == REFLECTIVE)
    {
      anArea += 2*aComp->lengthX*aComp->lengthZ; 
    }
  if(aComp->yzPlane == UNIPERIODIC || 
     aComp->yzPlane == REMOVE_UPPER ||
     aComp->yzPlane == REMOVE_LOWER)
    { 
      anArea += aComp->lengthY*aComp->lengthZ; 
    }
  else if(aComp->yzPlane == REFLECTIVE)
    {
      anArea += 2*aComp->lengthY*aComp->lengthZ; 
    }
  return anArea;
}

Point SpatiocyteStepper::coord2point(unsigned aCoord)
{
  unsigned aGlobalCol;
  unsigned aGlobalLayer;
  unsigned aGlobalRow;
  coord2global(aCoord, aGlobalRow, aGlobalLayer, aGlobalCol);
  //the center point of a voxel 
  Point aPoint;
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      aPoint.y = (aGlobalCol%2)*theHCPl+theHCPy*aGlobalLayer;
      aPoint.z = aGlobalRow*2*theNormalizedVoxelRadius+
        ((aGlobalLayer+aGlobalCol)%2)*theNormalizedVoxelRadius;
      aPoint.x = aGlobalCol*theHCPx;
      break;
    case CUBIC_LATTICE:
      aPoint.y = aGlobalLayer*2*theNormalizedVoxelRadius;
      aPoint.z = aGlobalRow*2*theNormalizedVoxelRadius;
      aPoint.x = aGlobalCol*2*theNormalizedVoxelRadius;
      break;
    }
  return aPoint;
};

double SpatiocyteStepper::getRowLength()
{
  return theNormalizedVoxelRadius*2;
}

double SpatiocyteStepper::getColLength()
{
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      return theHCPx;
    case CUBIC_LATTICE:
      return theNormalizedVoxelRadius*2;
    }
  return theNormalizedVoxelRadius*2;
}

double SpatiocyteStepper::getLayerLength()
{
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      return theHCPy;
    case CUBIC_LATTICE:
      return theNormalizedVoxelRadius*2;
    }
  return theNormalizedVoxelRadius*2;
}

double SpatiocyteStepper::getMinLatticeSpace()
{
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      return theHCPl;
    case CUBIC_LATTICE:
      return theNormalizedVoxelRadius*2;
    }
  return theNormalizedVoxelRadius*2;
}

unsigned SpatiocyteStepper::point2coord(Point& aPoint)
{
  unsigned aGlobalRow(0);
  unsigned aGlobalLayer(0);
  unsigned aGlobalCol(0);
  point2global(aPoint, aGlobalRow, aGlobalLayer, aGlobalCol);
  return global2coord(aGlobalRow, aGlobalLayer, aGlobalCol);
}

unsigned SpatiocyteStepper::global2coord(unsigned aGlobalRow,
                                             unsigned aGlobalLayer,
                                             unsigned aGlobalCol)
{
  return aGlobalRow+theRowSize*aGlobalLayer+theRowSize*theLayerSize*aGlobalCol;
}

void SpatiocyteStepper::point2global(Point aPoint, 
                                     unsigned& aGlobalRow,
                                     unsigned& aGlobalLayer,
                                     unsigned& aGlobalCol)
{
  double row(0);
  double layer(0);
  double col(0);
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      col = rint(aPoint.x/theHCPx);
      layer = rint((aPoint.y-(aGlobalCol%2)*theHCPl)/
                                        theHCPy);
      row = rint((aPoint.z-((aGlobalLayer+aGlobalCol)%2)*
          theNormalizedVoxelRadius)/(2*theNormalizedVoxelRadius));
      break;
    case CUBIC_LATTICE:
      col = rint(aPoint.x/(2*theNormalizedVoxelRadius));
      layer = rint(aPoint.y/(2*theNormalizedVoxelRadius));
      row = rint(aPoint.z/(2*theNormalizedVoxelRadius));
      break;
    }
  if(row < 0)
    {
      row = 0;
    }
  if(layer < 0)
    {
      layer = 0;
    }
  if(col < 0)
    {
      col = 0;
    }
  aGlobalRow = (unsigned)row;
  aGlobalLayer = (unsigned)layer;
  aGlobalCol = (unsigned)col;
  if(aGlobalCol >= theColSize)
    {
      aGlobalCol = theColSize-1;
    }
  if(aGlobalRow >= theRowSize)
    {
      aGlobalRow = theRowSize-1;
    }
  if(aGlobalLayer >= theLayerSize)
    {
      aGlobalLayer = theLayerSize-1;
    }
}

void SpatiocyteStepper::coord2global(unsigned aCoord,
                                     unsigned& aGlobalRow,
                                     unsigned& aGlobalLayer,
                                     unsigned& aGlobalCol) 
{
  aGlobalCol = (aCoord)/(theRowSize*theLayerSize);
  aGlobalLayer = ((aCoord)%(theRowSize*theLayerSize))/theRowSize;
  aGlobalRow = ((aCoord)%(theRowSize*theLayerSize))%theRowSize;
}

void SpatiocyteStepper::concatenateLayers(Voxel& aVoxel,
                                          unsigned a,
                                          unsigned aRow,
                                          unsigned aLayer,
                                          unsigned aCol)
{
  unsigned b(aRow+
                 theRowSize*aLayer+
                 theRowSize*theLayerSize*aCol);
  Voxel& anAdjoiningVoxel(theLattice[b]);
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      if((aLayer+1)%2+(aCol)%2 == 1)
        {
          aVoxel.adjoiningCoords[VENTRALN] = b;
          anAdjoiningVoxel.adjoiningCoords[DORSALS] = a;
          if(aRow < theRowSize-1)
            {
              unsigned c(aRow+1+ 
                             theRowSize*aLayer+
                             theRowSize*theLayerSize*aCol);
              Voxel& anAdjoiningVoxel(theLattice[c]);
              aVoxel.adjoiningCoords[VENTRALS] = c;
              anAdjoiningVoxel.adjoiningCoords[DORSALN] = a;
            }
        }
      else
        {
          aVoxel.adjoiningCoords[VENTRALS] = b;
          anAdjoiningVoxel.adjoiningCoords[DORSALN] = a;
          if(aRow > 0)
            {
              unsigned c(aRow-1+ 
                             theRowSize*aLayer+
                             theRowSize*theLayerSize*aCol);
              Voxel& anAdjoiningVoxel(theLattice[c]);
              aVoxel.adjoiningCoords[VENTRALN] = c;
              anAdjoiningVoxel.adjoiningCoords[DORSALS] = a;
            }
        }
      break;
    case CUBIC_LATTICE: 
      aVoxel.adjoiningCoords[VENTRAL] = b;
      anAdjoiningVoxel.adjoiningCoords[DORSAL] = a;
      break;
    }
}


void SpatiocyteStepper::concatenateCols(Voxel& aVoxel,
                                        unsigned a,
                                        unsigned aRow,
                                        unsigned aLayer,
                                        unsigned aCol)
{
  unsigned b(aRow+
                 theRowSize*aLayer+
                 theRowSize*theLayerSize*aCol);
  Voxel& anAdjoiningVoxel(theLattice[b]);
  switch(LatticeType)
    {
    case HCP_LATTICE: 
      if(aLayer%2 == 0)
        {
          if((aCol+1)%2 == 1)
            {
              aVoxel.adjoiningCoords[NW] = b;
              anAdjoiningVoxel.adjoiningCoords[SE] = a;
              if(aRow < theRowSize - 1)
                {
                  unsigned c(aRow+1+ 
                                 theRowSize*aLayer+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[SW] = c;
                  anAdjoiningVoxel.adjoiningCoords[NE] = a;
                }
              if(aLayer < theLayerSize-1)
                {
                  unsigned c(aRow+ 
                                 theRowSize*(aLayer+1)+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[WEST] = c;
                  anAdjoiningVoxel.adjoiningCoords[EAST] = a;
                }
            }
          else
            {
              aVoxel.adjoiningCoords[SW] = b;
              anAdjoiningVoxel.adjoiningCoords[NE] = a;
              if(aRow > 0)
                {
                  unsigned c(aRow-1+ 
                                 theRowSize*aLayer+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[NW] = c;
                  anAdjoiningVoxel.adjoiningCoords[SE] = a;
                }
              if(aLayer > 0)
                {
                  unsigned c(aRow+ 
                                 theRowSize*(aLayer-1)+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[WEST] = c;
                  anAdjoiningVoxel.adjoiningCoords[EAST] = a;
                }
            }
        }
      else
        {
          if((aCol+1)%2 == 1)
            {
              aVoxel.adjoiningCoords[SW] = b;
              anAdjoiningVoxel.adjoiningCoords[NE] = a;
              if(aRow > 0)
                {
                  unsigned c(aRow-1+ 
                                 theRowSize*aLayer+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[NW] = c;
                  anAdjoiningVoxel.adjoiningCoords[SE] = a;
                }
              if(aLayer < theLayerSize-1)
                {
                  unsigned c(aRow+ 
                                 theRowSize*(aLayer+1)+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[WEST] = c;
                  anAdjoiningVoxel.adjoiningCoords[EAST] = a;
                }
            }
          else
            {
              aVoxel.adjoiningCoords[NW] = b;
              anAdjoiningVoxel.adjoiningCoords[SE] = a;
              if(aRow < theRowSize - 1)
                {
                  unsigned c(aRow+1+ 
                                 theRowSize*aLayer+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[SW] = c;
                  anAdjoiningVoxel.adjoiningCoords[NE] = a;
                }
              if(aLayer > 0)
                {
                  unsigned c(aRow+
                                 theRowSize*(aLayer-1)+
                                 theRowSize*theLayerSize*aCol);
                  Voxel& anAdjoiningVoxel(theLattice[c]);
                  aVoxel.adjoiningCoords[WEST] = c;
                  anAdjoiningVoxel.adjoiningCoords[EAST] = a;
                }
            }
        }
      break;
    case CUBIC_LATTICE:
      aVoxel.adjoiningCoords[WEST] = b;
      anAdjoiningVoxel.adjoiningCoords[EAST] = a;
      break;
    }
}

void SpatiocyteStepper::concatenateRows(Voxel& aVoxel,
                                        unsigned a,
                                        unsigned aRow,
                                        unsigned aLayer,
                                        unsigned aCol)
{
  unsigned b(aRow+ 
                 theRowSize*aLayer+ 
                 theRowSize*theLayerSize*aCol);
  Voxel& anAdjoiningVoxel(theLattice[b]);
  aVoxel.adjoiningCoords[NORTH] = b;
  anAdjoiningVoxel.adjoiningCoords[SOUTH] = a;
}


void SpatiocyteStepper::concatenatePeriodicSurfaces()
{
  Comp* aRootComp(theComps[0]);
  for(unsigned i(0); i<=theRowSize*theLayerSize*theColSize-theRowSize;
      i+=theRowSize)
    {
      unsigned j(i+theRowSize-1);
      unsigned srcCoord(coord2row(i)+ 
                            theRowSize*coord2layer(i)+ 
                            theRowSize*theLayerSize*coord2col(i)); 
      unsigned destCoord(coord2row(j)+
                             theRowSize*coord2layer(i)+
                             theRowSize*theLayerSize*coord2col(i)); 
      if(aRootComp->xyPlane == UNIPERIODIC)
        { 
          replaceUniVoxel(srcCoord, destCoord);
        }
      else if(aRootComp->xyPlane == PERIODIC)
        { 
          replaceVoxel(srcCoord, destCoord);
        }
      else if(!isPeriodicEdge)
        {
          //We cannot have valid voxels pointing to itself it it is not periodic
          //to avoid incorrect homodimerization reaction. So we set such
          //molecules to null ID.
          theLattice[srcCoord].id = theNullID;
          theLattice[destCoord].id = theNullID;
        }
    }
  for(unsigned i(0); i<=theRowSize*theLayerSize*(theColSize-1)+theRowSize;)
    {
      unsigned j(theRowSize*(theLayerSize-1)+i);
      unsigned srcCoord(coord2row(i)+
                            theRowSize*coord2layer(i)+ 
                            theRowSize*theLayerSize*coord2col(i)); 
      unsigned destCoord(coord2row(i)+
                             theRowSize*coord2layer(j)+ 
                             theRowSize*theLayerSize*coord2col(i)); 
      if(aRootComp->xzPlane == UNIPERIODIC)
        { 
          replaceUniVoxel(srcCoord, destCoord);
        }
      else if(aRootComp->xzPlane == PERIODIC)
        { 
          replaceVoxel(srcCoord, destCoord);
        }
      else if(!isPeriodicEdge)
        {
          theLattice[srcCoord].id = theNullID;
          theLattice[destCoord].id = theNullID;
        }
      ++i;
      if(coord2layer(i) != 0)
        {
          i += (theLayerSize-1)*theRowSize;
        }
    }
  for(unsigned i(0); i!=theRowSize*theLayerSize; ++i)
    {
      unsigned srcCoord(coord2row(i)+
                            theRowSize*coord2layer(i)+ 
                            theRowSize*theLayerSize*0); 
      unsigned destCoord(coord2row(i)+
                             theRowSize*coord2layer(i)+ 
                             theRowSize*theLayerSize*(theColSize-1)); 
      if(aRootComp->yzPlane == UNIPERIODIC)
        { 
          replaceUniVoxel(srcCoord, destCoord);
        }
      else if(aRootComp->yzPlane == PERIODIC)
        { 
          replaceVoxel(srcCoord, destCoord);
        }
      else if(!isPeriodicEdge)
        {
          theLattice[srcCoord].id = theNullID;
          theLattice[destCoord].id = theNullID;
        }
    }
}

unsigned SpatiocyteStepper::coord2row(unsigned aCoord)
{
  return (aCoord%(theRowSize*theLayerSize))%theRowSize;
}

unsigned SpatiocyteStepper::coord2layer(unsigned aCoord)
{
  return (aCoord%(theRowSize*theLayerSize))/theRowSize;
}

unsigned SpatiocyteStepper::coord2col(unsigned aCoord)
{
  return aCoord/(theRowSize*theLayerSize);
}

void SpatiocyteStepper::replaceVoxel(unsigned src, unsigned dest)
{
  Voxel& aSrcVoxel(theLattice[src]);
  Voxel& aDestVoxel(theLattice[dest]);
  if(aSrcVoxel.id != theNullID && aDestVoxel.id != theNullID)
    {
      for(unsigned j(0); j!=theAdjoiningCoordSize; ++j)
        {
          if(aSrcVoxel.adjoiningCoords[j] == src &&
              aDestVoxel.adjoiningCoords[j] != dest)
            {
              aSrcVoxel.adjoiningCoords[j] = aDestVoxel.adjoiningCoords[j];
              for(unsigned k(0); k!=theAdjoiningCoordSize; ++k)
                {
                  Voxel& aVoxel(theLattice[aDestVoxel.adjoiningCoords[j]]);
                  if(aVoxel.adjoiningCoords[k] == dest)
                    {
                      aVoxel.adjoiningCoords[k] = src;
                    }
                }
            }
        }
      aDestVoxel.id = theNullID;
    }
}

void SpatiocyteStepper::replaceUniVoxel(unsigned src, unsigned dest)
{
  Voxel& aSrcVoxel(theLattice[src]);
  Voxel& aDestVoxel(theLattice[dest]);
  if(aSrcVoxel.id != theNullID && aDestVoxel.id != theNullID)
    {
      for(unsigned j(0); j!=theAdjoiningCoordSize; ++j)
        {
          if(aSrcVoxel.adjoiningCoords[j] == src)
            {
              for(unsigned k(0); k!=theAdjoiningCoordSize; ++k)
                {
                  Voxel& aVoxel(theLattice[aDestVoxel.adjoiningCoords[j]]);
                  if(aVoxel.adjoiningCoords[k] == dest)
                    {
                      aVoxel.adjoiningCoords[k] = src;
                    }
                }
            }
        }
      aDestVoxel.id = theNullID;
    }
}

void SpatiocyteStepper::shuffleAdjoiningCoords()
{
  for(std::vector<Voxel>::iterator i(theLattice.begin());
      i != theLattice.end(); ++i)
    {
      if((*i).id != theNullID)
        { 
          gsl_ran_shuffle(getRng(), (*i).adjoiningCoords, theAdjoiningCoordSize,
                          sizeof(unsigned));
        }
    }
}

void SpatiocyteStepper::setCompVoxelProperties()
{
  for(std::vector<Comp*>::iterator i(theComps.begin());
      i != theComps.end(); ++i)
    {
      switch ((*i)->dimension)
        {
        case 1:
          setLineCompProperties(*i);
          setLineVoxelProperties(*i);
          break;
        case 2:
          setSurfaceCompProperties(*i);
          setSurfaceVoxelProperties(*i);
          break;
        case 3:
        default:
          setVolumeCompProperties(*i);
        }
    }
}

void SpatiocyteStepper::setLineCompProperties(Comp* aComp)
{
    setSurfaceCompProperties(aComp);
}

void SpatiocyteStepper::setLineVoxelProperties(Comp* aComp)
{
    setSurfaceVoxelProperties(aComp);
}

void SpatiocyteStepper::setSurfaceCompProperties(Comp* aComp)
{
  aComp->vacantSpecies->removePeriodicEdgeVoxels();
  aComp->vacantSpecies->removeSurfaces();
  setDiffusiveComp(aComp);
}

void SpatiocyteStepper::setVolumeCompProperties(Comp* aComp)
{
  setDiffusiveComp(aComp);
}

void SpatiocyteStepper::setSurfaceVoxelProperties(Comp* aComp)
{
  if(!aComp->diffusiveComp)
    {
      Species* aVacantSpecies(aComp->vacantSpecies);
      for(unsigned i(0); i != aVacantSpecies->size(); ++i)
        {
          unsigned aCoord(aVacantSpecies->getCoord(i));
          optimizeSurfaceVoxel(aCoord, aComp);
          setSurfaceSubunit(aCoord, aComp);
        }
    }
}

void SpatiocyteStepper::setDiffusiveComp(Comp* aComp)
{
  FOR_ALL(System::Variables, aComp->system->getVariables())
    {
      Variable* aVariable(i->second);
      if(aVariable->getID() == "DIFFUSIVE")
        {
          String aStringID(aVariable->getName()); 
          aStringID = "System:" + aStringID;
          FullID aFullID(aStringID);
          System* aSystem(static_cast<System*>(getModel()->getEntity(aFullID)));
          aComp->diffusiveComp = system2Comp(aSystem);
        }
    }
  if(aComp->diffusiveComp)
    {
      Species* aVacantSpecies(aComp->vacantSpecies);
      for(unsigned i(0); i != aVacantSpecies->size(); ++i)
        {
          unsigned aCoord(aVacantSpecies->getCoord(i));
          aComp->diffusiveComp->vacantSpecies->addCompVoxel(aCoord);
        }
      aVacantSpecies->clearMolecules();
    }
}

void SpatiocyteStepper::optimizeSurfaceVoxel(unsigned aCoord, Comp* aComp)
{
  Voxel& aVoxel(theLattice[aCoord]);
  unsigned short surfaceID(aComp->vacantSpecies->getID());
  aComp->adjoinCount.resize(theAdjoiningCoordSize);
  aVoxel.surfaceCoords = new std::vector<std::vector<unsigned> >;
  aVoxel.surfaceCoords->resize(4);
  std::vector<unsigned>& immedSurface((*aVoxel.surfaceCoords)[IMMED]);
  std::vector<unsigned>& extSurface((*aVoxel.surfaceCoords)[EXTEND]);
  std::vector<unsigned>& innerVolume((*aVoxel.surfaceCoords)[INNER]);
  std::vector<unsigned>& outerVolume((*aVoxel.surfaceCoords)[OUTER]);
  std::vector<std::vector<unsigned> > sharedCoordsList;
  unsigned* forward(aVoxel.adjoiningCoords);
  unsigned* reverse(forward+theAdjoiningCoordSize);
  std::vector<unsigned> adjoiningCopy;
  for(unsigned k(0); k != theAdjoiningCoordSize; ++k)
    {
      adjoiningCopy.push_back(forward[k]);
    }
  //Separate adjoining surface voxels and adjoining volume voxels.
  //Put the adjoining surface voxels at the beginning of the
  //adjoiningCoords list while the volume voxels are put at the end:
  for(std::vector<unsigned>::iterator l(adjoiningCopy.begin());
      l != adjoiningCopy.end(); ++l)
    {
      if((*l) != aCoord && theLattice[*l].id != theNullID 
         && id2Comp(theLattice[*l].id)->dimension <= aComp->dimension)
        {
          ++aComp->adjoinCount[l-adjoiningCopy.begin()];
          (*forward) = (*l);
          ++forward;
          //immedSurface contains all adjoining surface voxels except the 
          //source voxel, aVoxel:
          immedSurface.push_back(*l);
          for(unsigned m(0); m != theAdjoiningCoordSize; ++m)
            {
              //extSurface contains the adjoining surface voxels of
              //adjoining surface voxels. They do not include the source voxel
              //and its adjoining voxels:
              unsigned extendedCoord(theLattice[*l].adjoiningCoords[m]);
              if(theLattice[extendedCoord].id == surfaceID &&
                 extendedCoord != aCoord &&
                 std::find(adjoiningCopy.begin(), adjoiningCopy.end(),
                      extendedCoord) == adjoiningCopy.end())
                {
                  std::vector<unsigned>::iterator n(std::find(
                      extSurface.begin(), extSurface.end(), extendedCoord));
                  if(n == extSurface.end())
                    {
                      extSurface.push_back(extendedCoord);
                      //We require shared immediate voxel which
                      //connects the extended voxel with the source voxel 
                      //for polymerization. Create a new list of shared
                      //immediate voxel each time a new extended voxel is added:
                      std::vector<unsigned> sharedCoords;
                      sharedCoords.push_back(*l);
                      sharedCoordsList.push_back(sharedCoords);
                    }
                  else
                    {
                      //An extended voxel may have multiple shared immediate
                      //voxels, so we insert the additional ones in the list:
                      sharedCoordsList[n-extSurface.begin()].push_back(*l);
                    }
                }
            }
        }
      else
        {
          --reverse;
          (*reverse) = (*l);
          //We know that it is not a surface voxel, so it would not
          //be a self-pointed adjoining voxel. If it is not inside the
          //surface (i.e., the parent volume) Comp, it must be an
          //outer volume voxel:
          if(!isInsideCoord(*l, aComp, 0))
            {
              outerVolume.push_back(*l);
            }
          //otherwise, it must be an inner volume voxel:
          else
            {
              innerVolume.push_back(*l);
            }
        }
    } 
  for(std::vector<std::vector<unsigned> >::iterator
      i(sharedCoordsList.begin()); i != sharedCoordsList.end(); ++i)
    {
      aVoxel.surfaceCoords->push_back(*i);
    }
  aVoxel.diffuseSize = forward-aVoxel.adjoiningCoords;
}

Species* SpatiocyteStepper::id2species(unsigned short id)
{
  return theSpecies[id];
}

unsigned short SpatiocyteStepper::getNullID()
{
  return theNullID;
}

Comp* SpatiocyteStepper::id2Comp(unsigned short id)
{
  return theSpecies[id]->getComp();
}

Comp* SpatiocyteStepper::system2Comp(System* aSystem)
{
  for(unsigned i(0); i != theComps.size(); ++i)
    {
      if(theComps[i]->system == aSystem)
        {
          return theComps[i];
        }
    }
  return NULL;
}

void SpatiocyteStepper::setSurfaceSubunit(unsigned aCoord, Comp* aComp)
{
  Voxel& aVoxel(theLattice[aCoord]);
  // The subunit is only useful for a cylindrical surface
  // and for polymerization on it.
  aVoxel.subunit = new Subunit;
  aVoxel.subunit->coord = aCoord;
  Point& aPoint(aVoxel.subunit->surfacePoint);
  aPoint = coord2point(aCoord);
  double aRadius(aComp->lengthY/2);
  Point aCenterPoint(aComp->centerPoint);
  Point aWestPoint(aComp->centerPoint);
  Point anEastPoint(aComp->centerPoint); 
  aWestPoint.x = aComp->centerPoint.x-aComp->lengthX/2+aComp->lengthY/2;
  anEastPoint.x = aComp->centerPoint.x+aComp->lengthX/2-aComp->lengthY/2;
  if(aPoint.x < aWestPoint.x)
    {
      aCenterPoint.x = aWestPoint.x;
    }
  else if(aPoint.x > anEastPoint.x)
    {
      aCenterPoint.x = anEastPoint.x;
    }
  else
    {
      aCenterPoint.x = aPoint.x;
    }
  double X(aPoint.x-aCenterPoint.x);
  double Y(aPoint.y-aCenterPoint.y);
  double Z(aPoint.z-aCenterPoint.z);
  double f(atan2(X, Z));
  double d(atan2(sqrt(X*X+Z*Z),Y));
  aPoint.x = aCenterPoint.x + sin(f)*aRadius*sin(d);
  aPoint.y = aCenterPoint.y + aRadius*cos(d);
  aPoint.z = aCenterPoint.z + cos(f)*aRadius*sin(d);
}

void SpatiocyteStepper::setIntersectingCompartmentList() 
{
  setIntersectingPeers();
  setIntersectingParent();
}

void SpatiocyteStepper::setIntersectingPeers()
{
  for(std::vector<Comp*>::reverse_iterator i(theComps.rbegin());
      i != theComps.rend(); ++i)
    {
      //Only proceed if the volume is not enclosed:
      if(!(*i)->system->isRootSystem() && (*i)->dimension == 3)
        {
          for(std::vector<Comp*>::reverse_iterator j(theComps.rbegin());
              j != theComps.rend(); ++j)
            {
              //Only proceed if the volume of the peer is enclosed:
              if((*i) != (*j) && !(*j)->system->isRootSystem() &&
                 (*j)->dimension == 3)
                {
                  //If i and j are peer volume compartments:
                  if((*i)->system->getSuperSystem() == 
                     (*j)->system->getSuperSystem())
                    {
                      Point a((*i)->centerPoint);
                      Point b((*j)->centerPoint);
                      if(((a.x+(*i)->lengthX/2 > b.x-(*j)->lengthX/2) ||
                          (a.x-(*i)->lengthX/2 < b.x+(*j)->lengthX/2)) && 
                         ((a.y+(*i)->lengthY/2 > b.y-(*j)->lengthY/2) ||
                          (a.y-(*i)->lengthY/2 < b.y+(*j)->lengthY/2)) &&
                         ((a.z+(*i)->lengthZ/2 > b.z-(*j)->lengthZ/2) ||
                          (a.z-(*i)->lengthZ/2 < b.z+(*j)->lengthZ/2)))
                        {
                          if((*i)->enclosed <= (*j)->enclosed) 
                            {
                              (*i)->intersectPeers.push_back(*j);
                            }
                          else
                            {
                              (*i)->intersectLowerPeers.push_back(*j);
                            }
                        }
                    }
                }
            }
        }
    }
}

void SpatiocyteStepper::setIntersectingParent() 
{
  for(std::vector<Comp*>::iterator i(theComps.begin());
      i != theComps.end(); ++i)
    {
      (*i)->isIntersectRoot = false;
      (*i)->isIntersectParent = false;
      //only proceed if the volume is not enclosed:
      if(!(*i)->system->isRootSystem() && (*i)->dimension == 3)
        {
          Comp* aParentComp(system2Comp((*i)->system->getSuperSystem())); 
          Point a((*i)->centerPoint);
          Point b(aParentComp->centerPoint);
          if((a.x+(*i)->lengthX/2 > b.x+aParentComp->lengthX/2) ||
             (a.x-(*i)->lengthX/2 < b.x-aParentComp->lengthX/2) || 
             (a.y+(*i)->lengthY/2 > b.y+aParentComp->lengthY/2) ||
             (a.y-(*i)->lengthY/2 < b.y-aParentComp->lengthY/2) ||
             (a.z+(*i)->lengthZ/2 > b.z+aParentComp->lengthZ/2) ||
             (a.z-(*i)->lengthZ/2 < b.z-aParentComp->lengthZ/2))
            {
              if(aParentComp->system->isRootSystem())
                {
                  (*i)->isIntersectRoot = true;
                }
              else
                {
                  (*i)->isIntersectParent = true;
                }
            }
        }
    }
}

void SpatiocyteStepper::compartmentalizeLattice() 
{
  for(unsigned i(0); i != theLattice.size(); ++i)
    {
      if(theLattice[i].id != theNullID)
        { 
          compartmentalizeVoxel(i, theComps[0]);
        }
    }
}

bool SpatiocyteStepper::compartmentalizeVoxel(unsigned aCoord, Comp* aComp)
{
  Voxel& aVoxel(theLattice[aCoord]);
  if(aComp->dimension == 3)
    {
      if(aComp->system->isRootSystem() ||
         isInsideCoord(aCoord, aComp, 0))
        {
          //Check if the voxel belongs to an intersecting peer compartment:
          if(!aComp->intersectPeers.empty())
            {
              //The function isPeerCoord also checks if the voxel is a future
              //surface voxel of the peer compartment: 
              if(isPeerCoord(aCoord, aComp))
                { 
                  return false;
                }
              if(aComp->surfaceSub && aComp->surfaceSub->enclosed)
                {
                  //Check if the voxel is neighbor of a peer voxel (either
                  //a future surface voxel)
                  if(isEnclosedSurfaceVoxel(aVoxel, aCoord, aComp))
                    {
                      aComp->surfaceSub->vacantSpecies->addCompVoxel(aCoord);
                      setMinMaxSurfaceDimensions(aCoord, aComp);
                      return true;
                    }
                }
            }
          if(aComp->isIntersectRoot)
            {
              Comp* aRootComp(system2Comp(aComp->system->getSuperSystem())); 
              if(aRootComp->surfaceSub && 
                 aRootComp->surfaceSub->enclosed && 
                 isRootSurfaceVoxel(aVoxel, aCoord, aRootComp))
                {
                  return false;
                }
              if(aComp->surfaceSub && 
                 isEnclosedRootSurfaceVoxel(aVoxel, aCoord, aComp, aRootComp))
                {
                  aComp->surfaceSub->vacantSpecies->addCompVoxel(aCoord);
                  setMinMaxSurfaceDimensions(aCoord, aComp);
                  return true;
                }
            }
          else if(aComp->isIntersectParent)
            {
              Comp* aParentComp(system2Comp(aComp->system->getSuperSystem())); 
              if(aComp->surfaceSub && aComp->surfaceSub->enclosed &&
                 isParentSurfaceVoxel(aVoxel, aCoord, aParentComp))
                {
                  aComp->surfaceSub->vacantSpecies->addCompVoxel(aCoord);
                  setMinMaxSurfaceDimensions(aCoord, aComp);
                  return true;
                }
            }
          for(unsigned i(0); i != aComp->immediateSubs.size(); ++i)
            {
              if(compartmentalizeVoxel(aCoord, aComp->immediateSubs[i]))
                {
                  return true;
                }
            }
          aComp->vacantSpecies->addCompVoxel(aCoord);
          return true;
        }
      if(aComp->surfaceSub)
        { 
          if(isInsideCoord(aCoord, aComp, 4) &&
             isSurfaceVoxel(aVoxel, aCoord, aComp))
            {
              aComp->surfaceSub->vacantSpecies->addCompVoxel(aCoord);
              setMinMaxSurfaceDimensions(aCoord, aComp);
              return true;
            }
        }
    }
  else if(aComp->system->getSuperSystem()->isRootSystem())
    {
      Comp* aRootComp(system2Comp(aComp->system->getSuperSystem())); 
      if(!isInsideCoord(aCoord, aRootComp, -4) &&
         isRootSurfaceVoxel(aVoxel, aCoord, aRootComp))
        {
          aComp->vacantSpecies->addCompVoxel(aCoord);
          setMinMaxSurfaceDimensions(aCoord, aRootComp);
          return true;
        }
    }
  return false;
}

bool SpatiocyteStepper::isRootSurfaceVoxel(Voxel& aVoxel, unsigned aCoord,
                                           Comp* aComp)
{
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      if(theLattice[aVoxel.adjoiningCoords[i]].id == theNullID ||
         aVoxel.adjoiningCoords[i] == aCoord)
        {
          return true;
        }
    }
  return false;
}

bool SpatiocyteStepper::isParentSurfaceVoxel(Voxel& aVoxel, unsigned aCoord,
                                             Comp* aComp)
{
  if(!isInsideCoord(aCoord, aComp, -4))
    {
      for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
        {
          if(!isInsideCoord(aVoxel.adjoiningCoords[i], aComp, 0))
            {
              return true;
            }
        }
    }
  return false;
}

bool SpatiocyteStepper::isEnclosedRootSurfaceVoxel(Voxel& aVoxel, 
                                                   unsigned aCoord,
                                                   Comp* aComp,
                                                   Comp* aRootComp)
{ 
  if(aComp->surfaceSub->enclosed || aRootComp->enclosed <= aComp->enclosed)
    { 
      if(!isInsideCoord(aCoord, aRootComp, -4))
        {
          if(!aRootComp->surfaceSub || 
             (aRootComp->surfaceSub && !aRootComp->surfaceSub->enclosed))
            {
              if(isRootSurfaceVoxel(aVoxel, aCoord, aComp))
                {
                  return true;
                }
            }
          else
            {
              for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
                {
                  unsigned adjoinCoord(aVoxel.adjoiningCoords[i]);
                  Voxel& adjoin(theLattice[adjoinCoord]);
                  if(isRootSurfaceVoxel(adjoin, adjoinCoord, aComp))
                    {
                      return true;
                    }
                }
            }
        }
    }
  return false;
}

bool SpatiocyteStepper::isSurfaceVoxel(Voxel& aVoxel, unsigned aCoord,
                                       Comp* aComp)
{
  if(isPeerCoord(aCoord, aComp))
    { 
      return false;
    }
  if(aComp->surfaceSub && !aComp->surfaceSub->enclosed &&
     isLowerPeerCoord(aCoord, aComp))
    {
      return false;
    }
  if(aComp->isIntersectRoot)
    {
      Comp* aRootComp(system2Comp(aComp->system->getSuperSystem())); 
      if(aRootComp->surfaceSub && aRootComp->surfaceSub->enclosed && 
         isRootSurfaceVoxel(aVoxel, aCoord, aRootComp))
        {
          return false;
        }
    }
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      if(isInsideCoord(aVoxel.adjoiningCoords[i], aComp, 0))
        {
          return true;
        }
    }
  return false;
}

bool SpatiocyteStepper::isEnclosedSurfaceVoxel(Voxel& aVoxel, 
                                               unsigned aCoord,
                                               Comp* aComp)
{
  for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
    {
      if(isPeerCoord(aVoxel.adjoiningCoords[i], aComp))
        {
          return true;
        }
    }
  return false;
}

bool SpatiocyteStepper::isLowerPeerCoord(unsigned aCoord, Comp* aComp)
{
  for(std::vector<Comp*>::iterator i(aComp->intersectLowerPeers.begin());
      i != aComp->intersectLowerPeers.end(); ++i)
    {
      if(isInsideCoord(aCoord, *i, 0))
        {
          return true;
        }
    }
  return false;
}

bool SpatiocyteStepper::isPeerCoord(unsigned aCoord, Comp* aComp)
{
  for(std::vector<Comp*>::iterator i(aComp->intersectPeers.begin());
      i != aComp->intersectPeers.end(); ++i)
    {
      if(isInsideCoord(aCoord, *i, 0))
        {
          return true;
        }
    }
  //The voxel is not inside one of the intersecting peer compartments,
  //we are now checking if the one of voxel's neighbor belongs to the
  //peer comp, to determine if aVoxel is a surface voxel of the peer:
  for(std::vector<Comp*>::iterator i(aComp->intersectPeers.begin());
      i != aComp->intersectPeers.end(); ++i)
    {
      if((*i)->surfaceSub && (*i)->surfaceSub->enclosed)
        { 
          if(isInsideCoord(aCoord, *i, 4))
            {
              for(unsigned j(0); j != theAdjoiningCoordSize; ++j)
                {
                  if(isInsideCoord(theLattice[aCoord].adjoiningCoords[j],
                                   *i, 0))
                    {
                      return true;
                    }
                }
            }
        }
    }
  return false;
}

bool SpatiocyteStepper::isLineVoxel(Voxel& aVoxel, unsigned aCoord,
                                    Comp* aComp)
{
    const double safety(2.0);
    const Point aPoint(coord2point(aCoord));

    double distance(aPoint.x - aComp->centerPoint.x);
    if (-safety < distance && distance <= 0)
    {
        // This is not efficient because we don't need to check volume voxels.
        // However, at this time, aVoxel.adjoinigVoxels is not properly 
        // aligned yet.
        for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
        {
            const Point adjoiningPoint(coord2point(aVoxel.adjoiningCoords[i]));
            const double distance_i(adjoiningPoint.x - aComp->centerPoint.x);
            if (distance_i > 0)
            {
                return true;
            }
        }
    }
                       
    distance = aPoint.y - aComp->centerPoint.y;
    if (-safety < distance && distance <= 0)
    {
        // This is not efficient because we don't need to check volume voxels.
        // However, at this time, aVoxel.adjoinigVoxels is not properly 
        // aligned yet.
        for(unsigned i(0); i != theAdjoiningCoordSize; ++i)
        {
            const Point adjoiningPoint(coord2point(aVoxel.adjoiningCoords[i]));
            const double distance_i(adjoiningPoint.y - aComp->centerPoint.y);
            if (distance_i > 0)
            {
                return true;
            }
        }
    }
    return false;
}

unsigned SpatiocyteStepper::getStartCoord()
{
  return 0;
}


void SpatiocyteStepper::setMinMaxSurfaceDimensions(unsigned aCoord, 
                                                   Comp* aComp)
{
  unsigned aRow;
  unsigned aLayer;
  unsigned aCol;
  coord2global(aCoord, aRow, aLayer, aCol);
  if(aRow < aComp->minRow)
    {
      aComp->minRow = aRow;
      aComp->surfaceSub->minRow = aRow;
    }
  else if(aRow > aComp->maxRow)
    {
      aComp->maxRow = aRow;
      aComp->surfaceSub->maxRow = aRow;
    }
  if(aCol < aComp->minCol)
    {
      aComp->minCol = aCol;
      aComp->surfaceSub->minCol = aCol;
    }
  else if(aCol > aComp->maxCol)
    {
      aComp->maxCol = aCol;
      aComp->surfaceSub->maxCol = aCol;
    }
  if(aLayer < aComp->minLayer)
    {
      aComp->minLayer = aLayer;
      aComp->surfaceSub->minLayer = aLayer;
    }
  else if(aLayer > aComp->maxLayer)
    {
      aComp->maxLayer = aLayer;
      aComp->surfaceSub->maxLayer = aLayer;
    }
}



void SpatiocyteStepper::rotateX(double angle, Point* aPoint, int sign)
{ 
  if(angle)
    {
      angle *= sign;
      double y(aPoint->y);
      double z(aPoint->z);
      aPoint->y = y*cos(angle)-z*sin(angle);
      aPoint->z = y*sin(angle)+z*cos(angle);
    }
}

void SpatiocyteStepper::rotateY(double angle, Point* aPoint, int sign)
{ 
  if(angle)
    {
      angle *= sign;
      double x(aPoint->x);
      double z(aPoint->z);
      aPoint->x = x*cos(angle)+z*sin(angle);
      aPoint->z = z*cos(angle)-x*sin(angle);
    }
}

void SpatiocyteStepper::rotateZ(double angle, Point* aPoint, int sign)
{ 
  if(angle)
    {
      angle *= sign;
      double x(aPoint->x);
      double y(aPoint->y);
      aPoint->x = x*cos(angle)-y*sin(angle);
      aPoint->y = x*sin(angle)+y*cos(angle);
    }
}

bool SpatiocyteStepper::isInsideCoord(unsigned aCoord,
                                      Comp* aComp, double delta)
{
  Point aPoint(coord2point(aCoord));
  Point aCenterPoint(aComp->centerPoint);
  Point aWestPoint(aComp->centerPoint);
  Point anEastPoint(aComp->centerPoint); 
  aPoint.x -= aCenterPoint.x;
  aPoint.y -= aCenterPoint.y;
  aPoint.z -= aCenterPoint.z;
  rotateX(aComp->rotateX, &aPoint);
  rotateY(aComp->rotateY, &aPoint);
  rotateZ(aComp->rotateZ, &aPoint);
  aPoint.x += aCenterPoint.x;
  aPoint.y += aCenterPoint.y;
  aPoint.z += aCenterPoint.z;
  double aRadius(0);
  switch(aComp->geometry)
    {
    case CUBOID:
      if(sqrt(pow(aPoint.x-aCenterPoint.x, 2)) <= 
         aComp->lengthX/2+theNormalizedVoxelRadius+delta &&
         sqrt(pow(aPoint.y-aCenterPoint.y, 2)) <= 
         aComp->lengthY/2+theNormalizedVoxelRadius+delta &&
         sqrt(pow(aPoint.z-aCenterPoint.z, 2)) <= 
         aComp->lengthZ/2+theNormalizedVoxelRadius+delta)
        {
          return true;
        }
      break;
    case ELLIPSOID:
      //If the distance between the voxel and the center point is less than 
      //or equal to radius-2, then the voxel cannot be a surface voxel:
      if(pow(aPoint.x-aCenterPoint.x, 2)/pow((aComp->lengthX+delta)/2, 2)+ 
         pow(aPoint.y-aCenterPoint.y, 2)/pow((aComp->lengthY+delta)/2, 2)+ 
         pow(aPoint.z-aCenterPoint.z, 2)/pow((aComp->lengthZ+delta)/2, 2) <= 1)
        {
          return true;
        }
      break;
    case CYLINDER: 
      //The axial point of the cylindrical portion of the rod:
      aCenterPoint.x = aPoint.x; 
      aWestPoint.x = aComp->centerPoint.x-aComp->lengthX/2;
      anEastPoint.x = aComp->centerPoint.x+aComp->lengthX/2;
      aRadius = aComp->lengthY/2+theNormalizedVoxelRadius;
      //If the distance between the voxel and the center point is less than 
      //or equal to the radius, then the voxel must be inside the Comp:
      if((aPoint.x >= aWestPoint.x && aPoint.x <= anEastPoint.x &&
          getDistance(&aPoint, &aCenterPoint) <= aRadius+delta))
        { 
          return true;
        }
      break;
    case ROD: 
      //The axial point of the cylindrical portion of the rod:
      aCenterPoint.x = aPoint.x; 
      aWestPoint.x = aComp->centerPoint.x-aComp->lengthX/2+aComp->lengthY/2;
      anEastPoint.x = aComp->centerPoint.x+aComp->lengthX/2-aComp->lengthY/2;
      aRadius = aComp->lengthY/2+theNormalizedVoxelRadius;
      //If the distance between the voxel and the center point is less than 
      //or equal to the radius, then the voxel must be inside the Comp:
      if((aPoint.x >= aWestPoint.x && aPoint.x <= anEastPoint.x &&
          getDistance(&aPoint, &aCenterPoint) <= aRadius+delta) ||
         (aPoint.x < aWestPoint.x &&
          getDistance(&aPoint, &aWestPoint) <= aRadius+delta) ||
         (aPoint.x > anEastPoint.x &&
          getDistance(&aPoint, &anEastPoint) <= aRadius+delta))
        { 
          return true;
        }
      break;
    case PYRAMID: 
      aRadius = ((aCenterPoint.y+aComp->lengthY/2)-aPoint.y)/aComp->lengthY;
      if(sqrt(pow(aPoint.y-aCenterPoint.y, 2)) <= aComp->lengthY/2+delta &&
         sqrt(pow(aPoint.x-aCenterPoint.x, 2)) <= 
         aComp->lengthX*aRadius/2+delta &&
         sqrt(pow(aPoint.z-aCenterPoint.z, 2)) <=
         aComp->lengthZ*aRadius/2+delta)
        {
          return true;
        }
      break;
    case ERYTHROCYTE: 
      if(delta > 0)
        {
          return true;
        }
      else if(delta < 0)
        {
          return false;
        }
      const double Rsq(pow(aPoint.x-aCenterPoint.x, 2)/
                       pow((aComp->lengthX)/2, 2)+ 
                       pow(aPoint.y-aCenterPoint.y, 2)/
                       pow((aComp->lengthY)/2, 2));
      if(Rsq > 1)
        {
          return false;
        }
      const double a(0.5);
      const double b(0.1);
      const double R(sqrt(Rsq));
      const double thickness(((1-cos(M_PI*0.5*R))*(a-b)+b)*sqrt(1-Rsq));
      const double height((aPoint.z-aCenterPoint.z)/(2*(aComp->lengthZ)));
      if(thickness*thickness >= height*height)
        {
          return true;
        }
      break;
    }
  return false;
}

void SpatiocyteStepper::populateComp(Comp* aComp)
{
  unsigned populationSize(0);
  std::vector<Species*> prioritySpecies;
  std::vector<Species*> diffusiveSpecies;
  std::vector<Species*> normalSpecies;
  for(std::vector<Species*>::const_iterator i(aComp->species.begin());
      i != aComp->species.end(); ++i)
    {
      if((*i)->getVacantSpecies()->getIsDiffusiveVacant())
        {
          diffusiveSpecies.push_back(*i);
        }
      else if((*i)->getIsPopulateSpecies())
        {
          populationSize += (unsigned)(*i)->getPopulateCoordSize();
          bool isPushed(false);
          std::vector<Species*> temp;
          std::vector<Species*>::const_iterator j(prioritySpecies.begin());
          while(j != prioritySpecies.end())
            {
              //Put high priority species and diffuse vacant species
              //in the high priority populate list
              if((*j)->getPopulatePriority() > (*i)->getPopulatePriority() ||
                 ((*j)->getPopulatePriority() == (*i)->getPopulatePriority() &&
                  (*j)->getIsDiffusiveVacant()))
                {
                  temp.push_back(*j);
                }
              else
                {
                  temp.push_back(*i); 
                  while(j != prioritySpecies.end())
                    {
                      temp.push_back(*j);
                      ++j;
                    }
                  isPushed = true;
                  break;
                }
              ++j;
            }
          if(!isPushed)
            {
              temp.push_back(*i);
            }
          prioritySpecies = temp;
        }
    }
  if(aComp->vacantSpecies->size() < populationSize)
    {
      THROW_EXCEPTION(ValueError, String(
                          getPropertyInterface().getClassName()) +
                          "There are " + int2str(populationSize) + 
                          " total molecules that must be uniformly " +
                          "populated,\nbut there are only "
                          + int2str(aComp->vacantSpecies->size()) + 
                          " vacant voxels of [" + 
                  aComp->vacantSpecies->getVariable()->getFullID().asString() +
                          "] that can be populated on.");
    }
  if(double(populationSize)/aComp->vacantSpecies->size() > 0.2)
    { 
      populateSpeciesDense(prioritySpecies, populationSize,
                           aComp->vacantSpecies->size());
    }
  else
    {
      populateSpeciesSparse(prioritySpecies);
    }
  for(std::vector<Species*>::const_iterator i(diffusiveSpecies.begin());
      i != diffusiveSpecies.end(); ++i)
    {
      (*i)->populateUniformOnDiffusiveVacant();
    }
}

void SpatiocyteStepper::populateSpeciesDense(std::vector<Species*>&
                                             aSpeciesList, unsigned aSize,
                                             unsigned availableVoxelSize)
{
  unsigned count(0);
  unsigned* populateVoxels(new unsigned[aSize]);
  unsigned* availableVoxels(new unsigned [availableVoxelSize]); 
  for(unsigned i(0); i != availableVoxelSize; ++i)
    {
      availableVoxels[i] = i;
    }
  gsl_ran_choose(getRng(), populateVoxels, aSize, availableVoxels,
                 availableVoxelSize, sizeof(unsigned));
  //gsl_ran_choose arranges the position ascending, so we need
  //to shuffle the order of voxel positions:
  gsl_ran_shuffle(getRng(), populateVoxels, aSize, sizeof(unsigned)); 
  for(std::vector<Species*>::const_iterator i(aSpeciesList.begin());
      i != aSpeciesList.end(); ++i)
    {
      (*i)->populateCompUniform(populateVoxels, &count);
    }
  delete[] populateVoxels;
  delete[] availableVoxels;
}

void SpatiocyteStepper::populateSpeciesSparse(std::vector<Species*>&
                                              aSpeciesList)
{
  for(std::vector<Species*>::const_iterator i(aSpeciesList.begin());
      i != aSpeciesList.end(); ++i)
    {
      (*i)->populateCompUniformSparse();
    }
}


void SpatiocyteStepper::clearComp(Comp* aComp)
{
  for(std::vector<Species*>::const_iterator i(aComp->species.begin());
      i != aComp->species.end(); ++i)
    {
      (*i)->removeMolecules();
      (*i)->updateMolecules();
    }
}


std::vector<Comp*> const& SpatiocyteStepper::getComps() const
{
  return theComps;
}

}
