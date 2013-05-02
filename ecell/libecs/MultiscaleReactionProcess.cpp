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

#include <MultiscaleReactionProcess.hpp>

LIBECS_DM_INIT(MultiscaleReactionProcess, Process); 


void MultiscaleReactionProcess::initializeThird()
{
  if(k == -1 && p == -1)
    {
      initializeMultiscaleWalkBindUnbind();
    }
  else
    {
      initializeMultiscaleCompReaction();
      DiffusionInfluencedReactionProcess::initializeThird();
    }
}

void MultiscaleReactionProcess::initializeMultiscaleWalkBindUnbind()
{ 
  //Set up the following:
  //theMultiscale = isMultiscale species 
  //M = a species on multiscale (isOnMultiscale)
  //N = a normal species that can bind with theMultiscaleSpecies to become
  //    M
  //This must be in initializeSecond or later since we need to know
  //if a species is multiscale, which is only set by the
  //CompartmentProcess in initializeFirst:
  if(A->getIsMultiscale())
    {
      theMultiscale = A;
      N = B;
    }
  else if(B->getIsMultiscale())
    {
      theMultiscale = B;
      N = A;
    }
  else
    {
      THROW_EXCEPTION(ValueError, String(
         getPropertyInterface().getClassName()) + " [" + 
          getFullID().asString() + "]: This process must have at least " +
         "one multiscale substrate species.");
    }
  if(N->getVacantSpecies() == theMultiscale)
    {
      THROW_EXCEPTION(ValueError, String(
         getPropertyInterface().getClassName()) + " [" + 
          getFullID().asString() + "]: The substrate " + 
          getIDString(N) + "'s vacant species is " +
          getIDString(theMultiscale) + " which is a multiscale species. " +
          "This reaction only expects the product's vacant species to be " +
          "a multiscale species. You should probably invert the " +
          "substrate with the product species to reverse the reaction.");
    }
  if(!D)
    {
      THROW_EXCEPTION(ValueError, String(
         getPropertyInterface().getClassName()) + " [" + 
          getFullID().asString() + "]: This process must have two " +
          "products.");
    }
  if(C->getIsMultiscale() && D->getIsOnMultiscale())
    {
      M = D;
    }
  else if(C->getIsOnMultiscale() && D->getIsMultiscale())
    {
      M = C;
    }
  else
    {
      THROW_EXCEPTION(ValueError, String(
         getPropertyInterface().getClassName()) + " [" + 
          getFullID().asString() + "]: This process must have at least " +
         "one product species on multiscale.");
    }
  //This must be set in
  //initializeThird since it requires vacant species properties
  //set by DiffusionProcess in initializeSecond:

  //If it is a dissociation reaction,
  //M diffuses on theMultiscale,
  //M unbinds from theMultiscale to become N:
  theMultiscale->setMultiscaleBindIDs(N->getID(), M->getID());
  theMultiscale->setMultiscaleUnbindIDs(M->getID(), N->getID());
  theMultiscale->setDiffusionInfluencedReaction(
        dynamic_cast<DiffusionInfluencedReactionProcess*>(this),
        N->getID(), 1); 
}

void MultiscaleReactionProcess::initializeMultiscaleCompReaction()
{
  if(!A->getIsMultiscaleComp() && !B->getIsMultiscaleComp())
    {
      THROW_EXCEPTION(ValueError, String(
             getPropertyInterface().getClassName()) + " [" + 
              getFullID().asString() + "]: This process must have at least " +
             "one multiscale substrate species or a substrate diffusing on " +
             "a multiscale species.");
    }
}

unsigned MultiscaleReactionProcess::getIdx(Species* aSpecies,
                                                    Voxel* mol,
                                                    const unsigned index)
{
  if(aSpecies->getIsOnMultiscale())
    {
      return aSpecies->getTag(index).vacantIdx;
    }
  return mol->idx;
}


//MuA + B -> [MuC <- MuA]
void MultiscaleReactionProcess::reactMuAtoMuC(Voxel* molA,
                                                       Voxel* molB,
                                                       const unsigned indexA,
                                                       const unsigned indexB)
{
  C->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  B->removeMolecule(indexB);
}
  

//A + MuB -> [MuC <- MuB]
void MultiscaleReactionProcess::reactMuBtoMuC(Voxel* molA,
                                                       Voxel* molB,
                                                       const unsigned indexA,
                                                       const unsigned indexB)
{
  C->addMoleculeInMulti(molB, getIdx(B, molB, indexB));
  A->removeMolecule(indexA);
}

//A + MuB -> [C <- molA] + [MuD <- MuB]
void MultiscaleReactionProcess::reactAtoC_MuBtoMuD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molA);
  A->softRemoveMolecule(indexA);
  //A != B, since only B is in multiscale comp:
  D->addMoleculeInMulti(molB, getIdx(B, molB, indexB));
  B->softRemoveMolecule(indexB);
}

//MuA + B -> [MuC <- MuA] + [D <- molB]
void MultiscaleReactionProcess::reactMuAtoMuC_BtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  A->softRemoveMolecule(indexA);
  D->addMolecule(molB);
  B->softRemoveMolecule(indexB);
}

//MuA + B -> [C <- molB] + [MuD <- MuA]
void MultiscaleReactionProcess::reactBtoC_MuAtoMuD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molB);
  B->softRemoveMolecule(indexB);
  D->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  A->softRemoveMolecule(indexA);
}

//A + MuB -> [MuC <- MuB] + [D <- molA]
void MultiscaleReactionProcess::reactMuBtoMuC_AtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMoleculeInMulti(molB, getIdx(B, molB, indexB));
  B->softRemoveMolecule(indexB);
  D->addMolecule(molA);
  A->softRemoveMolecule(indexA);
}
                  
//A + MuB -> [A == C] + [MuD <- MuB]
void MultiscaleReactionProcess::reactAeqC_MuBtoMuD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  D->addMoleculeInMulti(molB, getIdx(B, molB, indexB));
  B->softRemoveMolecule(indexB);
}

//A + MuB -> [MuA == MuC] + [MuD <- MuB]
void MultiscaleReactionProcess::reactMuAeqMuC_MuBtoMuD(Voxel* molA,
                                                       Voxel* molB,
                                                       const unsigned indexA,
                                                       const unsigned indexB)
{
  D->addMoleculeInMulti(molB, getIdx(B, molB, indexB));
  B->softRemoveMolecule(indexB);
}

//MuA + B -> [MuA == MuC] + [D <- molB]
void MultiscaleReactionProcess::reactMuAeqMuC_BtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  D->addMolecule(molB);
  B->softRemoveMolecule(indexB);
}

//MuA + B -> [B == C] + [MuD <- MuA]
void MultiscaleReactionProcess::reactBeqC_MuAtoMuD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  D->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  A->softRemoveMolecule(indexA);
}

//A + MuB -> [MuB == MuC] + [D <- molA]
void MultiscaleReactionProcess::reactMuBeqMuC_AtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  D->addMolecule(molA);
  A->softRemoveMolecule(indexA);
}

//A + MuB -> [MuC <- MuB] + [A == D]
void MultiscaleReactionProcess::reactMuBtoMuC_AeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMoleculeInMulti(molB, getIdx(B, molB, indexB));
  B->softRemoveMolecule(indexB);
}

//MuA + B -> [C <- molB] + [MuA == MuD]
void MultiscaleReactionProcess::reactBtoC_MuAeqMuD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molB);
  B->softRemoveMolecule(indexB);
}

//MuA + B -> [MuC <- MuA] + [B == D]
void MultiscaleReactionProcess::reactMuAtoMuC_BeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  A->softRemoveMolecule(indexA);
}

//MuA + B -> [MuC <- MuA] + [MuB == MuD]
void MultiscaleReactionProcess::reactMuAtoMuC_MuBeqMuD(Voxel* molA,
                                                       Voxel* molB,
                                                       const unsigned indexA,
                                                       const unsigned indexB)
{
  C->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  A->softRemoveMolecule(indexA);
}


//A + MuB -> [C <- molA] + [MuB == MuD]
void MultiscaleReactionProcess::reactAtoC_MuBeqMuD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molA);
  A->softRemoveMolecule(indexA);
}

//MuA + B -> [C <- molB]
void MultiscaleReactionProcess::reactBtoC_Multi(Voxel* molA,
                                                         Voxel* molB,
                                                         const unsigned indexA,
                                                         const unsigned indexB)
{
  C->addMolecule(molB);
  B->softRemoveMolecule(indexB);
  if(A->getIsOnMultiscale())
    {
      molA->idx = A->getTag(indexA).vacantIdx;
    }
  A->softRemoveMolecule(indexA);
}

//A + MuB -> [C <- molA]
void MultiscaleReactionProcess::reactAtoC_Multi(Voxel* molA,
                                                         Voxel* molB,
                                                         const unsigned indexA,
                                                         const unsigned indexB)
{
  C->addMolecule(molA);
  A->softRemoveMolecule(indexA);
  if(B->getIsOnMultiscale())
    {
      molB->idx = B->getTag(indexB).vacantIdx;
    }
  B->softRemoveMolecule(indexB);
}

//A + B -> [MuC <- MuA] + [MuD <- MuB]
void MultiscaleReactionProcess:: reactMuAtoMuC_MuBtoMuD(Voxel* molA,
                                                         Voxel* molB,
                                                         const unsigned indexA,
                                                         const unsigned indexB)
{
  const unsigned idxA(getIdx(A, molA, indexA));
  const unsigned idxB(getIdx(B, molB, indexB));
  A->softRemoveMolecule(indexA);
  removeMolecule(B, molB, indexB);
  C->addMoleculeInMulti(molA, idxA);
  D->addMoleculeInMulti(molB, idxB);
}

//A + B -> [MuB == MuC] + [MuD <- MuA]
void MultiscaleReactionProcess::reactMuBeqMuC_MuAtoMuD(Voxel* molA,
                                                         Voxel* molB,
                                                         const unsigned indexA,
                                                         const unsigned indexB)
{
  D->addMoleculeInMulti(molA, getIdx(A, molA, indexA));
  A->softRemoveMolecule(indexA);
}

void MultiscaleReactionProcess::setReactVarC_D()
{
  if(A == D)
    {
      //A + B -> variableC + [A == D]
      throwException("reactVarC_AeqD_Multi");
    }
  else if(B == D)
    {
      //A + B -> variableC + [B == D]
      throwException("reactVarC_BeqD_Multi");
    }
  else
    { 
      if(A->isReplaceable(D))
        {
          //A + B -> variableC + [D <- molA]
          throwException("reactVarC_AtoD_Multi");
        }
      else if(B->isReplaceable(D))
        {
          //A + B -> variableC + [D <- molB]
          throwException("reactVarC_BtoD_Multi");
        }
      else
        {
          //A + B -> variableC + [D <- molN]
          throwException("reactVarC_NtoD_Multi");
        }
    }
}

void MultiscaleReactionProcess::setReactVarD_C()
{
  if(A == C)
    {
      //A + B -> variableD + [A == C]
      throwException("reactVarD_AeqC_Multi");
    }
  else if(B == C)
    {
      //A + B -> variableD + [B == C]
      throwException("reactVarD_BeqC_Multi");
    }
  else
    { 
      if(A->isReplaceable(C))
        {
          //A + B -> variableD + [C <- molA]
          throwException("reactVarD_AtoC_Multi");
        }
      else if(B->isReplaceable(C))
        {
          //A + B -> variableD + [C <- molB]
          throwException("reactVarD_BtoC_Multi");
        }
      else
        {
          //A + B -> variableD + [C <- molN]
          throwException("reactVarD_NtoC_Multi");
        }
    }
}

void MultiscaleReactionProcess::setReactC()
{
  if(A == C)
    {
      //A + B -> [A == C]
      throwException("reactAeqC_Multi");
    }
  else if(B == C)
    {
      //A + B -> [B == C]
      throwException("reactBeqC_Multi");
    }
  else if(A->isReplaceable(C))
    {
      if(A->getIsMultiscaleComp() && !B->getIsMultiscaleComp())
        {
          //A + B -> [MuC <- MuA]
          reactM = &MultiscaleReactionProcess::reactMuAtoMuC;
        }
      else if(!A->getIsMultiscaleComp() && B->getIsMultiscaleComp())
        {
          //A + B -> [C <- molA]
          reactM = &MultiscaleReactionProcess::reactAtoC_Multi;
        }
      else
        {
          //A + B -> [MuC <- MuA]
          throwException("reactMuAtoMuC");
        }
    }
  else if(B->isReplaceable(C))
    {
      if(B->getIsMultiscaleComp() && !A->getIsMultiscaleComp())
        {
          //A + B -> [MuC <- MuB]
          reactM = &MultiscaleReactionProcess::reactMuBtoMuC;
        }
      else if(!B->getIsMultiscaleComp() && A->getIsMultiscaleComp())
        {
          //A + B -> [C <- molB]
          reactM = &MultiscaleReactionProcess::reactBtoC_Multi;
        }
      else
        {
          //A + B -> [MuC <- MuB]
          throwException("reactMuBtoMuC");
        }
    }
  else
    {
      //A + B -> [C <- molN]
      throwException("reactNtoC_Multi");
    }
}

void MultiscaleReactionProcess::setReactD()
{
  if(A == C && B == D)
    {
      //A + B -> [A == C] + [B == D]
      reactM = &MultiscaleReactionProcess::reactNone;
    }
  else if(B == C && A == D)
    {
      //A + B -> [B == C] + [A == D]
      reactM = &MultiscaleReactionProcess::reactNone;
    }
  else if(A == C)
    {
      if(B->isReplaceable(D))
        {
          if(B->getIsMultiscaleComp() && !A->getIsMultiscaleComp())
            {
              //A + B -> [A == C] + [MuD <- MuB]
              reactM = &MultiscaleReactionProcess::reactAeqC_MuBtoMuD;
            }
          else if(!B->getIsMultiscaleComp() && A->getIsMultiscaleComp())
            {
              //A + B -> [MuA == MuC] + [D <- molB]
              reactM = &MultiscaleReactionProcess::reactMuAeqMuC_BtoD;
            }
          else
            {
              //A + B -> [MuA == MuC] + [MuD <- MuB]
              reactM = &MultiscaleReactionProcess::reactMuAeqMuC_MuBtoMuD;
            }
        }
      else
        {
          //A + B -> [A == C] + [D <- molN]
          throwException("reactAeqC_NtoD_Multi");
        }
    }
  else if(B == C)
    {
      if(A->isReplaceable(D))
        {
          if(A->getIsMultiscaleComp() && !B->getIsMultiscaleComp())
            {
              //A + B -> [B == C] + [MuD <- MuA]
              reactM = &MultiscaleReactionProcess::reactBeqC_MuAtoMuD;
            }
          else if(!A->getIsMultiscaleComp() && B->getIsMultiscaleComp())
            {
              //A + B -> [MuB == MuC] + [D <- molA]
              reactM = &MultiscaleReactionProcess::reactMuBeqMuC_AtoD;
            }
          else
            {
              //A + B -> [MuB == MuC] + [MuD <- MuA]
              reactM = &MultiscaleReactionProcess::reactMuBeqMuC_MuAtoMuD;
            }
        }
      else
        {
          //A + B -> [B == C] + [D <- molN]
          throwException("reactBeqC_NtoD_Multi");
        }
    }
  else if(A == D)
    {
      if(B->isReplaceable(C))
        {
          if(B->getIsMultiscaleComp() && !A->getIsMultiscaleComp())
            {
              //A + B -> [MuC <- MuB] + [A == D]
              reactM = &MultiscaleReactionProcess::reactMuBtoMuC_AeqD;
            }
          else if(!B->getIsMultiscaleComp() && A->getIsMultiscaleComp())
            {
              //A + B -> [C <- molB] + [MuA == MuD] 
              reactM = &MultiscaleReactionProcess::reactBtoC_MuAeqMuD;
            }
          else
            {
              //A + B -> [MuC <- MuB] + [MuA == MuD]
              throwException("reactMuBtoMuC_MuAeqMuD");
            }
        }
      else
        {
          //A + B -> [C <- molN] + [A == D]
          throwException("reactNtoC_AeqD_Multi");
        }
    }
  else if(B == D)
    {
      if(A->isReplaceable(C))
        {
          if(A->getIsMultiscaleComp() && !B->getIsMultiscaleComp())
            {
              //A + B -> [MuC <- MuA] + [B == D]
              reactM = &MultiscaleReactionProcess::reactMuAtoMuC_BeqD;
            }
          else if(!A->getIsMultiscaleComp() && B->getIsMultiscaleComp())
            {
              //A + B -> [C <- molA] + [MuB == MuD] 
              reactM = &MultiscaleReactionProcess::reactAtoC_MuBeqMuD;
            }
          else
            {
              //A + B -> [MuC <- MuA] + [MuB == MuD] 
              reactM = &MultiscaleReactionProcess::reactMuAtoMuC_MuBeqMuD;
            }
        }
      else
        {
          //A + B -> [C <- molN] + [B == D]
          throwException("reactNtoC_BeqD_Multi");
        }
    }
  else
    {
      if(A->isReplaceable(C))
        {
          if(B->isReplaceable(D))
            {
              if(B->getIsMultiscaleComp() && !A->getIsMultiscaleComp())
                {
                  //A + B -> [C <- molA] + [MuD <- MuB]
                  reactM = &MultiscaleReactionProcess::reactAtoC_MuBtoMuD;
                }
              else if(!B->getIsMultiscaleComp() && A->getIsMultiscaleComp())
                {
                  //A + B -> [MuC <- MuA] + [D <- molB]
                  reactM = &MultiscaleReactionProcess::reactMuAtoMuC_BtoD;
                }
              else
                {
                  //A + B -> [MuC <- MuA] + [MuD <- MuB]
                  reactM = &MultiscaleReactionProcess::
                    reactMuAtoMuC_MuBtoMuD;
                }
            }
          else
            {
              //A + B -> [C <- molA] + [D <- molN]
              throwException("reactAtoC_NtoD_Multi");
            }
        }
      else if(B->isReplaceable(C))
        {
          if(A->isReplaceable(D))
            {
              if(A->getIsMultiscaleComp() && !B->getIsMultiscaleComp())
                {
                  //A + B -> [C <- molB] + [MuD <- MuA]
                  reactM = &MultiscaleReactionProcess::reactBtoC_MuAtoMuD;
                }
              else if(!A->getIsMultiscaleComp() && B->getIsMultiscaleComp())
                {
                  //A + B -> [MuC <- MuB] + [D <- molA]
                  reactM = &MultiscaleReactionProcess::reactMuBtoMuC_AtoD;
                }
              else
                {
                  //A + B -> [MuC <- MuB] + [MuD <- MuA]
                  throwException("reactMuBtoMuC_MuAtoMuD");
                }
            }
          else
            {
              //A + B -> [C <- molB] + [D <- molN]
              throwException("reactBtoC_NtoD_Multi");
            }
        }
      else
        {
          //A + B -> [C <- molN] + [D <- molN]
          throwException("reactNtoC_NtoD_Multi");
        }
    }
}

void MultiscaleReactionProcess::setReactMethod()
{
  if(variableC && D)
    {
      setReactVarC_D();
    }
  else if(variableD && C)
    {
      setReactVarD_C();
    }
  else if(variableC)
    {
      if(variableD)
        {
          //A + B -> variableC + variableD
          throwException("reactVarC_VarD_Multi");
        }
      else
        {
          //A + B -> variableC
          throwException("reactVarC_Multi");
        }
    }
  else if(D)
    {
      setReactD();
    }
  else
    {
      setReactC();
    }
}



