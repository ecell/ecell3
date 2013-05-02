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


#ifndef __DiffusionProcess_hpp
#define __DiffusionProcess_hpp

#include <sstream>
#include <SpatiocyteProcess.hpp>
#include <SpatiocyteSpecies.hpp>

LIBECS_DM_CLASS(DiffusionProcess, SpatiocyteProcess)
{ 
  typedef void (DiffusionProcess::*WalkMethod)(void) const;
public:
  LIBECS_DM_OBJECT(DiffusionProcess, Process)
    {
      INHERIT_PROPERTIES(Process);
      PROPERTYSLOT_SET_GET(Integer, Origins);
      PROPERTYSLOT_SET_GET(Integer, RegularLattice);
      PROPERTYSLOT_SET_GET(Real, D);
      PROPERTYSLOT_SET_GET(Real, Interval);
      PROPERTYSLOT_SET_GET(Real, P);
      PROPERTYSLOT_SET_GET(Real, Propensity);
      PROPERTYSLOT_SET_GET(Real, WalkProbability);
    }
  DiffusionProcess():
    Origins(0),
    RegularLattice(0),
    D(0),
    Interval(0),
    P(1),
    Propensity(0),
    WalkProbability(1),
    theDiffusionSpecies(NULL),
    theVacantSpecies(NULL),
    theWalkMethod(&DiffusionProcess::walk) {}
  virtual ~DiffusionProcess() {}
  SIMPLE_SET_GET_METHOD(Integer, Origins);
  SIMPLE_SET_GET_METHOD(Integer, RegularLattice);
  SIMPLE_SET_GET_METHOD(Real, D);
  SIMPLE_SET_GET_METHOD(Real, Interval);
  SIMPLE_SET_GET_METHOD(Real, P);
  SIMPLE_SET_GET_METHOD(Real, Propensity);
  SIMPLE_SET_GET_METHOD(Real, WalkProbability);
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      SpatiocyteProcess::initialize();
      isPriorityQueued = true;
      for(VariableReferenceVector::iterator
          i(theVariableReferenceVector.begin());
          i != theVariableReferenceVector.end(); ++i)
        {
          Species* aSpecies(theSpatiocyteStepper->variable2species(
                                   (*i).getVariable())); 
          if(!(*i).getCoefficient())
            {
              if(theDiffusionSpecies)
                {
                  THROW_EXCEPTION(ValueError, String(
                                  getPropertyInterface().getClassName()) +
                                  "[" + getFullID().asString() + 
                                  "]: A DiffusionProcess requires only one " +
                                  "nonHD variable reference with zero " +
                                  "coefficient as the species to be " +
                                  "diffused, but " + 
                                  getIDString(theDiffusionSpecies) + " and " +
                                  getIDString(aSpecies) + " are given."); 
                }
              theDiffusionSpecies = aSpecies;
              theDiffusionSpecies->setDiffusionCoefficient(D);
            }
          else
            {
              if(theVacantSpecies)
                {
                  THROW_EXCEPTION(ValueError, String(
                                  getPropertyInterface().getClassName()) +
                                  "[" + getFullID().asString() + 
                                  "]: A DiffusionProcess requires only one " +
                                  "nonHD variable reference with negative " +
                                  "coefficient as the vacant species to be " +
                                  "diffused on, but " +
                                  getIDString(theVacantSpecies) + " and " +
                                  getIDString(aSpecies) + " are given."); 
                }
              theVacantSpecies = aSpecies;
            }
        }
      if(!theDiffusionSpecies)
        {
          THROW_EXCEPTION(ValueError, String(
                          getPropertyInterface().getClassName()) +
                          "[" + getFullID().asString() + 
                          "]: A DiffusionProcess requires only one " +
                          "nonHD variable reference with zero coefficient " +
                          "as the species to be diffused, but none is given."); 
        }
    }
  virtual void initializeSecond()
    {
      SpatiocyteProcess::initializeSecond();
      if(theVacantSpecies)
        {
          if(!theVacantSpecies->getIsMultiscale())
            {
              theVacantSpecies->setIsDiffusiveVacant();
            }
          theDiffusionSpecies->setVacantSpecies(theVacantSpecies);
        }
    }
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
          //From 4rv^2 = 2lDt, alpha = 4/(2l), where l is the dimension:
          double alpha(2); //default value for 1D diffusion
          if(theDiffusionSpecies->getDimension() == 2)
            {
              if(RegularLattice || theDiffusionSpecies->getIsRegularLattice())
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
      else if(Interval > 0)
        {
          theInterval = Interval;
        }
      theDiffusionSpecies->setDiffusionInterval(theInterval);
      if(Origins)
        {
          theDiffusionSpecies->initMoleculeOrigins();
        }
      if(theDiffusionSpecies->getIsDiffusiveVacant())
        {
          theWalkMethod = &DiffusionProcess::walkVacant;
        }
      else if(theDiffusionSpecies->getIsMultiscale())
        {
          if(Propensity)
            {
              theDiffusionSpecies->setWalkPropensity(Propensity);
              if(theDiffusionSpecies->getIsRegularLattice())
                {
                  if(Origins)
                    {
                      theWalkMethod = 
                      &DiffusionProcess::walkMultiscalePropensityRegularOrigins;
                    }
                  else
                    {
                      theWalkMethod = 
                        &DiffusionProcess::walkMultiscalePropensityRegular;
                    }
                }
              else
                {
                  theWalkMethod = &DiffusionProcess::walkMultiscalePropensity;
                }
            }
          else
            {
              if(theDiffusionSpecies->getIsRegularLattice())
                {
                  if(Origins)
                    {
                      theWalkMethod = 
                        &DiffusionProcess::walkMultiscaleRegularOrigins;
                    }
                  else
                    {
                      theWalkMethod = &DiffusionProcess::walkMultiscaleRegular;
                    }
                }
              else
                {
                  theWalkMethod = &DiffusionProcess::walkMultiscale;
                }
            }
        }
      else 
        {
          if(theDiffusionSpecies->getIsRegularLattice())
            {
              if(theDiffusionSpecies->getIsOnMultiscale())
                {
                  theWalkMethod = &DiffusionProcess::walkOnMultiscaleRegular;
                }
              else
                {
                  theWalkMethod = &DiffusionProcess::walkRegular;
                }
            }
          else
            {
              theWalkMethod = &DiffusionProcess::walk;
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
      (this->*theWalkMethod)();
      theDiffusionSpecies->finalizeReactions();
    }
  void walk() const
    {
      theDiffusionSpecies->walk();
    }
  void walkRegular() const
    {
      theDiffusionSpecies->walkRegular();
    }
   void walkOnMultiscaleRegular() const
    {
      theDiffusionSpecies->walkOnMultiscaleRegular();
    }
  void walkVacant() const
    {
      theDiffusionSpecies->walkVacant();
    }
  void walkMultiscale() const
    {
      theDiffusionSpecies->walkMultiscale();
    }
  void walkMultiscalePropensity() const
    {
      theDiffusionSpecies->walkMultiscalePropensity();
    }
  void walkMultiscaleRegular() const
    {
      theDiffusionSpecies->walkMultiscaleRegular();
    }
  void walkMultiscaleRegularOrigins() const
    {
      theDiffusionSpecies->walkMultiscaleRegularOrigins();
    }
  void walkMultiscalePropensityRegular() const
    {
      theDiffusionSpecies->walkMultiscalePropensityRegular();
    }
  void walkMultiscalePropensityRegularOrigins() const
    {
      theDiffusionSpecies->walkMultiscalePropensityRegularOrigins();
    }
  virtual void initializeLastOnce()
    {
      //theDiffusionSpecies->addInterruptedProcess(this);
    }
  /*
  virtual double getInterval()
    {
      if(theDiffusionSpecies->size())
        {
          return theInterval;
        }
      return libecs::INF;
    }
    */
protected:
  unsigned Origins;
  unsigned RegularLattice;
  double D;
  double Interval;
  double P;
  double Propensity;
  double WalkProbability;
  Species* theDiffusionSpecies;
  Species* theVacantSpecies;
  WalkMethod theWalkMethod;
};

#endif /* __DiffusionProcess_hpp */

