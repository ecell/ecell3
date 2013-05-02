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


#ifndef __DiffusionInfluencedReactionProcess_hpp
#define __DiffusionInfluencedReactionProcess_hpp

#include <ReactionProcess.hpp>
#include <DiffusionInfluencedReactionProcessInterface.hpp>

LIBECS_DM_CLASS_EXTRA_1(DiffusionInfluencedReactionProcess, ReactionProcess, virtual DiffusionInfluencedReactionProcessInterface)
{ 
  typedef void (DiffusionInfluencedReactionProcess::*Method)(Voxel*, Voxel*,
                                       const unsigned, const unsigned);
public:
  LIBECS_DM_OBJECT(DiffusionInfluencedReactionProcess, Process)
    {
      INHERIT_PROPERTIES(ReactionProcess);
      PROPERTYSLOT_SET_GET(Integer, Collision);
    }
  SIMPLE_SET_GET_METHOD(Integer, Collision);
  DiffusionInfluencedReactionProcess():
    Collision(0) {}
  virtual ~DiffusionInfluencedReactionProcess() {}
  virtual void addSubstrateInterrupt(Species* aSpecies, Voxel* aMolecule) {}
  virtual void removeSubstrateInterrupt(Species* aSpecies, Voxel* aMolecule) {}
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      ReactionProcess::initialize();
      if(getOrder() != 2)
        {
          THROW_EXCEPTION(ValueError, 
                          String(getPropertyInterface().getClassName()) + 
                          "[" + getFullID().asString() + 
                          "]: Only second order scheme is allowed for "+
                          "diffusion influenced reactions.");
        }
      checkSubstrates();
    }
  virtual void checkSubstrates();
  virtual void initializeSecond();
  virtual void initializeThird();
  virtual void react(Voxel* molA, Voxel* molB, const unsigned indexA,
                     const unsigned indexB)
    {
      moleculeA = molA;
      moleculeB = molB;
      interruptProcessesPre();
      (this->*reactM)(molA, molB, indexA, indexB);
      interruptProcessesPost();
    }
  virtual void bind(Voxel*, const unsigned) {}
  virtual void unbind(Voxel*) {}
  virtual void finalizeReaction()
    {
      //The number of molecules may have changed for both reactant and product
      //species. We need to update SpatiocyteNextReactionProcesses which are
      //dependent on these species:
      for(unsigned i(0); i != theInterruptedProcesses.size(); ++i)
        { 
          theSpatiocyteStepper->addInterruptedProcess(
                                                theInterruptedProcesses[i]);
        }
    }
  virtual void printParameters();
  virtual void setReactMethod();
protected:
  void calculateReactionProbability();
  void throwException(String);
  void addMoleculeF();
  void removeMolecule(Species*, Voxel*, const unsigned) const;
  void removeMolecule(Species*, Voxel*, const unsigned, Species*) const;
  Voxel* getPopulatableVoxel(Species*, Voxel*, Voxel*);
  Voxel* getPopulatableVoxel(Species*, Voxel*, Voxel*, Voxel*);
  void reactNone(Voxel*, Voxel*, const unsigned, const unsigned) {}
  void reactVarC_AeqD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarC_BeqD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarC_AtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarC_BtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarC_NtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarD_AeqC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarD_BeqC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarD_AtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarD_BtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarD_NtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAeqC_BtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAeqC_NtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBeqC_AtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBeqC_NtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBtoC_AeqD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactNtoC_AeqD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAtoC_BeqD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAtoC_BeqD_tagAtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactNtoC_BeqD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAtoC_BtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAtoC_NtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBtoC_AtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBtoC_NtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactNtoC_NtoD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarC_VarD(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAtoC_compNtoE(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactVarC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAeqC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBeqC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactAtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactNtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBtoC_tagBtoC(Voxel*, Voxel*, const unsigned, const unsigned);
  void reactBtoC_tagAtoC(Voxel*, Voxel*, const unsigned, const unsigned);
protected:
  unsigned Collision;
  double D_A;
  double D_B;
  double r_v;
  double V;
  Method reactM;
};

#endif /* __DiffusionInfluencedReactionProcess_hpp */
