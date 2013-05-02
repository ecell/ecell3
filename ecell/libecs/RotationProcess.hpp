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


#ifndef __RotationProcess_hpp
#define __RotationProcess_hpp

#include <sstream>
#include <libecs/MethodProxy.hpp>
#include <DiffusionProcess.hpp>
#include <SpatiocyteProcess.hpp>
#include <SpatiocyteSpecies.hpp>

LIBECS_DM_CLASS(RotationProcess, DiffusionProcess)
{ 
  typedef void (RotationProcess::*RotateMethod)(void) const;
public:
  LIBECS_DM_OBJECT(RotationProcess, Process)
    {
      INHERIT_PROPERTIES(DiffusionProcess);
    }
  virtual ~RotationProcess() {}
  virtual void initializeFourth()
    {
      double rho(theDiffusionSpecies->getMaxReactionProbability());
      if(rho > P)
        {
          WalkProbability = P/rho;
        }
      theDiffusionSpecies->rescaleReactionProbabilities(WalkProbability);
      if(D > 0)
        {
          double r_v(theDiffusionSpecies->getDiffuseRadius());
          double alpha(2); //default for 1D diffusion
          if(theDiffusionSpecies->getDimension() == 2)
            {
              if(theDiffusionSpecies->getIsRegularLattice())
                {
                  alpha = 1;
                }
              else
                {
                  alpha = pow((2*sqrt(2)+4*sqrt(3)+3*sqrt(6)+sqrt(22))/
                              (6*sqrt(2)+4*sqrt(3)+3*sqrt(6)), 2);
                }
            }
          else if(theDiffusionSpecies->getDimension() == 3)
            {
              alpha = 2.0/3;
            }
          theInterval = alpha*r_v*r_v*WalkProbability/D;
        }
      theDiffusionSpecies->setDiffusionInterval(theInterval);
      if(theDiffusionSpecies->getIsMultiscale())
        {
          if(Propensity)
            {
              if(theDiffusionSpecies->getIsRegularLattice())
                {
                  theRotateMethod = 
                    &RotationProcess::rotateMultiscalePropensityRegular;
                }
              else
                {
                  //theRotateMethod = 
                  //&RotationProcess::rotateMultiscalePropensity;
                }
            }
          else
            {
              if(theDiffusionSpecies->getIsRegularLattice())
                {
                  theRotateMethod = &RotationProcess::rotateMultiscaleRegular;
                }
              else
                {
                  //theRotateMethod = &RotationProcess::walkMultiscale;
                }
            }
        }
      //After initializeFourth, this process will be enqueued in the priority
      //queue, so we must update the number of molecules of the diffusion 
      //species if it is a diffusiveVacant species:
      theDiffusionSpecies->updateMoleculeSize();
    }
  virtual void printParameters()
    {
      String aProcess(String(getPropertyInterface().getClassName()) + 
                      "[" + getFullID().asString() + "]");
      cout << aProcess << std::endl;
      cout << "  " << getIDString(theDiffusionSpecies) << " ";
      cout << ":" << std::endl << "  Diffusion interval=" <<
        theInterval << ", D=" << D << ", Walk probability (P/rho)=" <<
        WalkProbability << std::endl;
    }
  virtual void fire()
    {
      requeue();
      theDiffusionSpecies->resetFinalizeReactions();
      (this->*theRotateMethod)();
      theDiffusionSpecies->finalizeReactions();
    }
  void rotateMultiscaleRegular() const
    {
      theDiffusionSpecies->rotateMultiscaleRegular();
    }
  void rotateMultiscalePropensityRegular() const
    {
      theDiffusionSpecies->rotateMultiscalePropensityRegular();
    }
private:
  RotateMethod theRotateMethod;
};

#endif /* __RotationProcess_hpp */

