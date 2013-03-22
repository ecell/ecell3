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

#include "PolymerFragmentationProcess.hpp"
#include "SpatiocyteSpecies.hpp"

LIBECS_DM_INIT(PolymerFragmentationProcess, Process);

void PolymerFragmentationProcess::fire()
{
  //polyA + polyB -> (poly)C + (poly)D
  if(A && B && C && D)
    {
      //A must be the reference subunit:
      int reactantIndex(gsl_rng_uniform_int(getStepper()->getRng(),
                                             theReactantSize));
      Voxel* moleculeA(theReactants[reactantIndex]);
      Voxel* moleculeC;
      if(A->getVacantID() != C->getVacantID())
        {
          moleculeC = C->getRandomAdjoiningVoxel(moleculeA, SearchVacant);
          //Only proceed if we can find an adjoining vacant voxel
          //of A which can be occupied by C:
          if(moleculeC == NULL)
            {
              return;
            }
        }
      else
        {
          moleculeC = moleculeA;
        }
      Voxel* moleculeB(moleculeA->subunit->targetVoxels[theBendIndexA]);
      Voxel* moleculeD;
      if(B->getVacantID() != D->getVacantID())
        {
          //Find an adjoining vacant voxel of B belonging to the
          //compartment of D which is not molecule C:
          moleculeD = D->getRandomAdjoiningVoxel(moleculeB, moleculeC,
                                                 SearchVacant);
          //Only proceed if we can find an adjoining vacant voxel
          //of B which can be occupied by D and the vacant voxel is not
          //used by moleculeC:
          if(moleculeD == NULL)
            {
              return;
            }
        }
      else
        {
          moleculeD = moleculeB;
        }
      //When we remove the molecule A, this process' queue in priority
      //queue will be updated since the substrate number has changed:
      A->removeMolecule(moleculeA);
      C->addMolecule(moleculeC);
      B->removeMolecule(moleculeB);
      D->addMolecule(moleculeD);
      //Change the connections only after updating the queues of
      //other depolymerize processes since the connections are used by
      //the processes when checking:
      if(C->getIsPolymer())
        {
          thePolymerizeProcess->removeContPoint(moleculeB->subunit,
                            &moleculeA->subunit->targetPoints[theBendIndexA]);
          moleculeA->subunit->targetVoxels[theBendIndexA] = NULL;
        }
      else
        {
          thePolymerizeProcess->resetSubunit(moleculeA->subunit);
        }
      if(D->getIsPolymer())
        {
          moleculeB->subunit->sourceVoxels[theBendIndexB] = NULL;
        }
      else
        {
          thePolymerizeProcess->resetSubunit(moleculeB->subunit);
        }
    }
  //polyA -> C + D
  else if(A && !B && C && D)
    {
      //A must be the reference subunit:
      int reactantIndex(gsl_rng_uniform_int(getStepper()->getRng(),
                                             theReactantSize));
      Voxel* moleculeA(theReactants[reactantIndex]);
      Voxel* moleculeC;
      Voxel* moleculeD;
      if(A->getVacantID() != C->getVacantID())
        {
          moleculeC = C->getRandomAdjoiningVoxel(moleculeA, SearchVacant);
          //Only proceed if we can find an adjoining vacant voxel
          //of A which can be occupied by C:
          if(moleculeC == NULL)
            {
              return;
            }
          if(A->getVacantID() != D->getVacantID())
            {
              //Find an adjoining vacant voxel of A belonging to the
              //compartment of D which is not molecule C:
              moleculeD = D->getRandomAdjoiningVoxel(moleculeA, moleculeC,
                                                     SearchVacant);
              //Only proceed if we can find an adjoining vacant voxel
              //of B which can be occupied by D and the vacant voxel is not
              //used by moleculeC:
              if(moleculeD == NULL)
                {
                  moleculeD = D->getRandomAdjoiningVoxel(moleculeC,
                                                         SearchVacant);
                  if(moleculeD == NULL)
                    {
                      return;
                    }
                }
            }
          else
            {
              moleculeD = moleculeA;
            }
        }
      else
        {
          moleculeC = moleculeA;
          moleculeD = D->getRandomAdjoiningVoxel(moleculeA, SearchVacant);
          if(moleculeD == NULL)
            {
              return;
            }
        }
      //When we remove the molecule A, this process' queue in priority
      //queue will be updated since the substrate number has changed:
      A->removeMolecule(moleculeA);
      C->addMolecule(moleculeC);
      D->addMolecule(moleculeD);
      //Change the connections only after updating the queues of
      //other depolymerize processes since the connections are used by
      //the processes when checking:
      thePolymerizeProcess->resetSubunit(moleculeA->subunit);
    }
  for(std::vector<SpatiocyteProcess*>::const_iterator 
      i(theInterruptingProcesses.begin());
      i!=theInterruptingProcesses.end(); ++i)
    {
      (*i)->substrateValueChanged(theSpatiocyteStepper->getCurrentTime());
    }
}

void PolymerFragmentationProcess::addSubstrateInterrupt(Species* aSpecies,
                                                        Voxel* aMolecule)
{
  //If first order:
  if(!B)
    {
      addMoleculeA(aMolecule);
      return;
    }
  //If second order:
  else if(aSpecies == A)
    {
      Subunit* subunitA(aMolecule->subunit);
      if(subunitA->targetVoxels.size() &&
         subunitA->targetVoxels[theBendIndexA] &&
         subunitA->targetVoxels[theBendIndexA]->id == B->getID())
        { 
          addMoleculeA(aMolecule);
          return;
        }
    }
  if(aSpecies == B)
    {
      Subunit* subunitB(aMolecule->subunit);
      if(subunitB->sourceVoxels.size() &&
         subunitB->sourceVoxels[theBendIndexB] &&
         subunitB->sourceVoxels[theBendIndexB]->id == A->getID())
        { 
          addMoleculeA(subunitB->sourceVoxels[theBendIndexB]);
          return;
        }
    }
}

void PolymerFragmentationProcess::removeSubstrateInterrupt(Species* aSpecies,
                                                           Voxel* aMolecule)
{
  if(aSpecies == A)
    {
      for(unsigned int i(0); i < theReactantSize; ++i)
        {
          if(aMolecule == theReactants[i])
            {
              theReactants[i] = theReactants[--theReactantSize];
              substrateValueChanged(theSpatiocyteStepper->getCurrentTime());
              return;
            }
        }
    }
  if(aSpecies == B)
    {
      Subunit* subunitB(aMolecule->subunit);
      Voxel* moleculeA(subunitB->sourceVoxels[theBendIndexB]);
      if(subunitB->sourceVoxels.size() && moleculeA &&
         moleculeA->id == A->getID())
        { 
          for(unsigned int i(0); i < theReactantSize; ++i)
            {
              if(moleculeA == theReactants[i])
                {
                  theReactants[i] = theReactants[--theReactantSize];
                  substrateValueChanged(theSpatiocyteStepper->getCurrentTime());
                  return;
                }
            }
        }
    }
}

void PolymerFragmentationProcess::initializeLastOnce()
{
  A->addInterruptedProcess(this);
  theBendIndexA = A->getBendIndex(BendAngle);
  if(B)
    {
      if(A != B)
        {
          B->addInterruptedProcess(this);
        }
      theBendIndexB = B->getBendIndex(BendAngle);
    }
}


