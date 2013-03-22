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
// based on GillespieProcess.hpp
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//


#ifndef __PolymerFragmentationProcess_hpp
#define __PolymerFragmentationProcess_hpp

#include <libecs/MethodProxy.hpp>
#include "ReactionProcess.hpp"
#include "PolymerizationProcess.hpp"

class PolymerizationProcess;

LIBECS_DM_CLASS(PolymerFragmentationProcess, ReactionProcess)
{ 
public:
  LIBECS_DM_OBJECT(PolymerFragmentationProcess, Process)
    {
      INHERIT_PROPERTIES(ReactionProcess);
      PROPERTYSLOT_SET_GET(Real, BendAngle);
    }
  SIMPLE_SET_GET_METHOD(Real, BendAngle);
  PolymerFragmentationProcess():
    theReactantSize(0),
    BendAngle(0) {}
  virtual ~PolymerFragmentationProcess() {}
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      ReactionProcess::initialize();
      isPriorityQueued = true;
      if(getOrder() != 1 && getOrder() != 2)
        {
          THROW_EXCEPTION(ValueError, 
                          String(getPropertyInterface().getClassName()) + 
                          "[" + getFullID().asString() + 
                          "]: This PolymerFragmentationProcess requires " +
                          "two substrates.");
        }
      if(p == -1)
        {
          p = k;
        }
      else
        {
          k = p;
        }
    }
  virtual void initializeLastOnce();
  virtual bool isContinuous() const
    {
      return true;
    }
  virtual GET_METHOD(Real, StepInterval)
    {
      return getPropensity()*
        (-log(gsl_rng_uniform_pos(getStepper()->getRng())));
    }
  virtual void fire();
  virtual void addSubstrateInterrupt(Species* aSpecies, Voxel* aMolecule);
  virtual void removeSubstrateInterrupt(Species* aSpecies, Voxel* aMolecule);
  void addMoleculeA(Voxel* aMolecule)
    { 
      ++theReactantSize;
      if(theReactantSize > theReactants.size())
        {
          theReactants.push_back(aMolecule);
        }
      else
        {
          theReactants[theReactantSize-1] = aMolecule;
        }
      substrateValueChanged(theSpatiocyteStepper->getCurrentTime());
    }
  void setPolymerizeProcess(PolymerizationProcess* aProcess)
    {
      thePolymerizeProcess = aProcess;
    }
protected:
  double getPropensity() const
    {
      if(theReactantSize > 0 && p > 0)
        {
          return 1/(p*theReactantSize);
        }
      else
        {
          return libecs::INF;
        }
    }
protected:
  unsigned int theReactantSize;
  int theBendIndexA;
  int theBendIndexB;
  double BendAngle;
  PolymerizationProcess* thePolymerizeProcess;
  std::vector<Voxel*> theReactants;
};


#endif /* __PolymerFragmentationProcess_hpp */


