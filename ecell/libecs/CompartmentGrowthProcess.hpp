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


#ifndef __CompartmentGrowthProcess_hpp
#define __CompartmentGrowthProcess_hpp

#include <sstream>
#include "SpatiocyteProcess.hpp"

LIBECS_DM_CLASS(CompartmentGrowthProcess, SpatiocyteProcess)
{ 
public:
  LIBECS_DM_OBJECT(CompartmentGrowthProcess, Process)
    {
      INHERIT_PROPERTIES(Process);
      PROPERTYSLOT_SET_GET(Real, k);
      PROPERTYSLOT_SET_GET(Integer, Axis);
    }
  CompartmentGrowthProcess():
    k(1),
    Axis(1) {}
  virtual ~CompartmentGrowthProcess() {}
  SIMPLE_SET_GET_METHOD(Real, k);
  SIMPLE_SET_GET_METHOD(Integer, Axis);
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      SpatiocyteProcess::initialize();
      isPriorityQueued = true;
    }
  virtual void initializeFifth()
    {
      theComp = theSpatiocyteStepper->system2Comp(getSuperSystem());
      theStepInterval = 1;  
    }
  virtual void printParameters()
    {
      std::cout << "k:" << k << std::endl;
      std::cout << "Axis:" << Axis << std::endl;
    }
  virtual void fire()
    {
      int col(gsl_rng_uniform_int(getStepper()->getRng(),
                                  theComp->maxCol-theComp->minCol-4));
      col += theComp->minCol;
      theSpatiocyteStepper->growCompartment(theComp, Axis, col);
      theTime += theStepInterval;
      thePriorityQueue->moveTop();
    }
  virtual void initializeLastOnce()
    {
    }
protected:
  Comp* theComp;
  double k;
  int Axis;
};

#endif /* __CompartmentGrowthProcess_hpp */
