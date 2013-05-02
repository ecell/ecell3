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


#ifndef __SpatiocyteNextReactionProcess_hpp
#define __SpatiocyteNextReactionProcess_hpp

#include <sstream>
#include <ReactionProcess.hpp>

LIBECS_DM_CLASS(SpatiocyteNextReactionProcess, ReactionProcess)
{ 
  typedef double (SpatiocyteNextReactionProcess::*PropensityMethod)(void); 
public:
  LIBECS_DM_OBJECT(SpatiocyteNextReactionProcess, Process)
    {
      INHERIT_PROPERTIES(ReactionProcess);
      PROPERTYSLOT_SET_GET(Real, SpaceA);
      PROPERTYSLOT_SET_GET(Real, SpaceB);
      PROPERTYSLOT_SET_GET(Real, SpaceC);
      PROPERTYSLOT_SET_GET(Integer, Deoligomerize);
      PROPERTYSLOT_SET_GET(Integer, BindingSite);
      PROPERTYSLOT_SET_GET(Integer, ImplicitUnbind);
      PROPERTYSLOT_SET_GET(Polymorph, Rates);
    }
  SpatiocyteNextReactionProcess():
    Deoligomerize(0),
    theDeoligomerIndex(0),
    BindingSite(-1),
    ImplicitUnbind(0),
    initSizeA(0),
    initSizeB(0),
    initSizeC(0),
    initSizeD(0),
    SpaceA(0),
    SpaceB(0),
    SpaceC(0),
    thePropensityMethod(&SpatiocyteNextReactionProcess::
                               getPropensityZerothOrder) {}
  virtual ~SpatiocyteNextReactionProcess() {}
  SIMPLE_SET_GET_METHOD(Real, SpaceA);
  SIMPLE_SET_GET_METHOD(Real, SpaceB);
  SIMPLE_SET_GET_METHOD(Real, SpaceC);
  SIMPLE_SET_GET_METHOD(Integer, Deoligomerize);
  SIMPLE_SET_GET_METHOD(Integer, BindingSite);
  SIMPLE_SET_GET_METHOD(Integer, ImplicitUnbind);
  SIMPLE_GET_METHOD(Polymorph, Rates);
  void setRates(const Polymorph& aValue)
    {
      Rates = aValue;
      PolymorphVector aValueVector(aValue.as<PolymorphVector>());
      for(unsigned i(0); i != aValueVector.size(); ++i)
        {
          theRates.push_back(aValueVector[i].as<double>());
        }
    }
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      ReactionProcess::initialize();
      isPriorityQueued = true;
      checkExternStepperInterrupted();
      if(!(getOrder() == 0 || getOrder() == 1 || getOrder() == 2))
        {
          if(getZeroVariableReferenceOffset() > 2)
            {
              THROW_EXCEPTION(ValueError, 
                              String(getPropertyInterface().getClassName()) + 
                              "[" + getFullID().asString() + 
                              "]: Only zeroth, first or second order scheme " + 
                              "is allowed.");
            }
        }
      if(variableA)
        {
          initSizeA = variableA->getValue();
        }
      if(variableB)
        {
          initSizeB = variableB->getValue();
        }
      if(variableC)
        {
          initSizeC = variableC->getValue();
        }
      if(variableD)
        {
          initSizeD = variableD->getValue();
        }
      if(Deoligomerize && theRates.size() && theRates.size() != Deoligomerize)
        {
          THROW_EXCEPTION(ValueError, 
                          String(getPropertyInterface().getClassName()) + 
                          "[" + getFullID().asString() + 
                          "]: For deoligomerize reactions, the number of " +
                          "rates given in Rates must match the number of " +
                          "binding sites specified by Deoligomerize.");
        }
    }
  virtual void preinitialize();
  virtual void initializeSecond();
  virtual void initializeThird();
  virtual bool isContinuous() 
    {
      return true;
    }
  virtual double getNewInterval();
  virtual double getInterval(double);
  virtual void fire();
  virtual void initializeFourth();
  virtual void printParameters();
  virtual bool isDependentOn(const Process*) const;
  virtual bool react();
  virtual double getPropensity() const;
  virtual double getNewPropensity();
protected:
  void updateSubstrates();
  void updateMoleculesA();
  double getIntervalUnbindAB();
  double getIntervalUnbindMultiAB();
  virtual void calculateOrder();
  virtual bool reactACD(Species*, Species*, Species*);
  virtual bool reactAC(Species*, Species*);
  virtual bool reactDeoligomerize(Species*, Species*);
  virtual bool reactACbind(Species*, Species*);
  virtual bool reactACDbind(Species*, Species*, Species*);
  virtual void reactABCD();
  virtual void reactABC();
  virtual bool reactMultiABC();
  virtual Voxel* reactvAC(Variable*, Species*);
  virtual Comp* getComp2D(Species*);
  virtual Voxel* reactvAvBC(Species*);
  double getPropensityZerothOrder(); 
  double getPropensityFirstOrder();
  double getPropensityFirstOrderDeoligomerize();
  double getPropensitySecondOrderHomo(); 
  double getPropensitySecondOrderHetero(); 
  void removeMoleculeE();
  void checkExternStepperInterrupted();
  void setVariableReferences(const VariableReferenceVector&);
  void setDeoligomerIndex(const unsigned);
protected:
  unsigned Deoligomerize;
  unsigned nextIndexA;
  unsigned theDeoligomerIndex;
  int BindingSite;
  int ImplicitUnbind;
  double initSizeA;
  double initSizeB;
  double initSizeC;
  double initSizeD;
  double SpaceA;
  double SpaceB;
  double SpaceC;
  double thePropensity;
  std::vector<double> theRates;
  Polymorph Rates;
  std::stringstream pFormula;
  PropensityMethod thePropensityMethod;  
  std::vector<Voxel*> moleculesA;
};

#endif /* __SpatiocyteNextReactionProcess_hpp */
