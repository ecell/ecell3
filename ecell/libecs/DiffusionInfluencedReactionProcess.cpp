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

#include <DiffusionInfluencedReactionProcess.hpp>
#include <SpatiocyteSpecies.hpp>

LIBECS_DM_INIT(DiffusionInfluencedReactionProcess, Process);

void DiffusionInfluencedReactionProcess::checkSubstrates()
{
  //HD_A or HD_B:
	if(variableA)
    {
      THROW_EXCEPTION(ValueError, String(
        getPropertyInterface().getClassName()) + " [" + getFullID().asString() +
		    "]: This process cannot have a HD substrate species: " + 
        getIDString(variableA));
    }
  if(variableB)
    {
      THROW_EXCEPTION(ValueError, String(
        getPropertyInterface().getClassName()) + " [" + getFullID().asString() +
		    "]: This process cannot have a HD substrate species: " + 
        getIDString(variableB));
    }
}

void DiffusionInfluencedReactionProcess::initializeSecond()
{
  ReactionProcess::initializeSecond(); 
  A->setCollision(Collision);
  B->setCollision(Collision);
}

void DiffusionInfluencedReactionProcess::initializeThird()
{
  ReactionProcess::initializeThird();
  A->setDiffusionInfluencedReactantPair(B); 
  B->setDiffusionInfluencedReactantPair(A); 
  r_v = theSpatiocyteStepper->getVoxelRadius();
  D_A = A->getDiffusionCoefficient();
  D_B = B->getDiffusionCoefficient();
  calculateReactionProbability();
  if(A->getIsDiffusing())
    {
      A->setDiffusionInfluencedReaction(this, B->getID(), p); 
    }
  if(B->getIsDiffusing())
    {
      B->setDiffusionInfluencedReaction(this, A->getID(), p); 
    }
  setReactMethod();
}

void DiffusionInfluencedReactionProcess::removeMolecule(Species* aSpecies, 
                                                  Voxel* mol,
                                                  const unsigned index) const
{
  if(A != B)
    {
      aSpecies->removeMolecule(index);
    }
  else
    {
      //If A == B, indexB is no longer valid after molA is removed,
      //so need to use the current index to remove molB:
      aSpecies->removeMolecule(mol);
    }
}

void DiffusionInfluencedReactionProcess::removeMolecule(Species* substrate, 
                                                        Voxel* mol,
                                                        const unsigned index,
                                                        Species* product) const
{
  if(A != B)
    { 
      product->addMolecule(mol, substrate->getTag(index));
      substrate->softRemoveMolecule(index);
    }
  else
    {
      Tag& aTag(substrate->getTag(mol));
      substrate->softRemoveMolecule(mol);
      product->addMolecule(mol, aTag);
    }
}

Voxel* DiffusionInfluencedReactionProcess::getPopulatableVoxel(
                                                             Species* aSpecies,
                                                             Voxel* molA,
                                                             Voxel* molB)
{
  Voxel* mol(aSpecies->getRandomAdjoiningVoxel(molA, SearchVacant));
  if(!mol)
    {
      mol = aSpecies->getRandomAdjoiningVoxel(molB, SearchVacant);
    }
  return mol;
}

Voxel* DiffusionInfluencedReactionProcess::getPopulatableVoxel(
                                                             Species* aSpecies,
                                                             Voxel* molA,
                                                             Voxel* molB,
                                                             Voxel* molC)
{
  Voxel* mol(aSpecies->getRandomAdjoiningVoxel(molA, molC, SearchVacant));
  if(!mol)
    {
      mol = aSpecies->getRandomAdjoiningVoxel(molB, molC, SearchVacant);
    }
  return mol;
}

//A + B -> variableC + [D <- molA]
void DiffusionInfluencedReactionProcess::reactVarC_AtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableC->addValue(1);
  D->addMolecule(molA, A->getTag(indexA));
  A->softRemoveMolecule(indexA);
  removeMolecule(B, molB, indexB);
}

//A + B -> variableC + [D <- molB]
void DiffusionInfluencedReactionProcess::reactVarC_BtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableC->addValue(1);
  D->addMolecule(molB, B->getTag(indexB));
  B->softRemoveMolecule(indexB);
  removeMolecule(A, molA, indexA);
}

//A + B -> variableC + [D <- molN]
void DiffusionInfluencedReactionProcess::reactVarC_NtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{ 
  Voxel* mol(getPopulatableVoxel(D, molA, molB));
  if(mol)
    {
      variableC->addValue(1);
      //TODO: need to use the correct tag here:
      D->addMolecule(mol, A->getTag(indexA));
      A->removeMolecule(indexA);
      removeMolecule(B, molB, indexB);
    }
}

//A + B -> variableC + [D == molA]
void DiffusionInfluencedReactionProcess::reactVarC_AeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableC->addValue(1);
  B->removeMolecule(indexB);
}

//A + B -> variableC + [D == molB]
void DiffusionInfluencedReactionProcess::reactVarC_BeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableC->addValue(1);
  A->removeMolecule(indexA);
}

//A + B -> variableD + [C <- molA]
void DiffusionInfluencedReactionProcess::reactVarD_AtoC(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableD->addValue(1);
  C->addMolecule(molA, A->getTag(indexA));
  A->softRemoveMolecule(indexA);
  removeMolecule(B, molB, indexB);
}

//A + B -> variableD + [C <- molB]
void DiffusionInfluencedReactionProcess::reactVarD_BtoC(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableD->addValue(1);
  C->addMolecule(molB, B->getTag(indexB));
  B->softRemoveMolecule(indexB);
  removeMolecule(A, molA, indexA);
}

//A + B -> variableD + [C <- molN]
void DiffusionInfluencedReactionProcess::reactVarD_NtoC(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{ 
  Voxel* mol(getPopulatableVoxel(C, molA, molB));
  if(mol)
    {
      variableD->addValue(1);
      //TODO: need to use the correct tag here:
      C->addMolecule(mol, A->getTag(indexA));
      A->removeMolecule(indexA);
      removeMolecule(B, molB, indexB);
    }
}


//A + B -> variableD + [C == A]
void DiffusionInfluencedReactionProcess::reactVarD_AeqC(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableD->addValue(1);
  B->removeMolecule(indexB);
}

//A + B -> variableD + [C == B]
void DiffusionInfluencedReactionProcess::reactVarD_BeqC(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableD->addValue(1);
  A->removeMolecule(indexA);
}

//A + B -> variableC + variableD
void DiffusionInfluencedReactionProcess::reactVarC_VarD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  variableC->addValue(1);
  variableD->addValue(1);
  A->removeMolecule(indexA);
  removeMolecule(B, molB, indexB);
}

//A + B -> variableC
void DiffusionInfluencedReactionProcess::reactVarC(Voxel* molA,
                                                   Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  variableC->addValue(1);
  A->removeMolecule(indexA);
  removeMolecule(B, molB, indexB);
}

//A + B -> [A == C] + [D <- molB]
void DiffusionInfluencedReactionProcess::reactAeqC_BtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  D->addMolecule(molB, B->getTag(indexB));
  B->softRemoveMolecule(indexB);
}

//A + B -> [A == C] + [D <- molN]
void DiffusionInfluencedReactionProcess::reactAeqC_NtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(D, molA, molB));
  if(mol)
    {
      D->addMolecule(mol, B->getTag(indexB));
      B->removeMolecule(indexB);
    }
}

//A + B -> [B == C] + [D <- molA]
void DiffusionInfluencedReactionProcess::reactBeqC_AtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  D->addMolecule(molA, A->getTag(indexA));
  A->softRemoveMolecule(indexA);
}

//A + B -> [B == C] + [D <- molN]
void DiffusionInfluencedReactionProcess::reactBeqC_NtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(D, molA, molB));
  if(mol)
    {
      D->addMolecule(mol, A->getTag(indexA));
      A->removeMolecule(indexA);
    }
}

//A + B -> [C <- molB] + [A == D]
void DiffusionInfluencedReactionProcess::reactBtoC_AeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molB, B->getTag(indexB));
  B->softRemoveMolecule(indexB);
}

//A + B -> [C <- molN] + [A == D]
void DiffusionInfluencedReactionProcess::reactNtoC_AeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(C, molA, molB));
  if(mol)
    {
      C->addMolecule(mol, B->getTag(indexB));
      B->removeMolecule(indexB);
    }
}

//A + B -> [C <- molA] + [B == D]
void DiffusionInfluencedReactionProcess::reactAtoC_BeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molA);
  A->softRemoveMolecule(indexA);
}


//A + B -> [C <- molA] + [tagC <- tagA] + [B == D]
void DiffusionInfluencedReactionProcess::reactAtoC_BeqD_tagAtoC(Voxel* molA,
                                                                Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molA, A->getTag(indexA));
  A->softRemoveMolecule(indexA);
}

//A + B -> [C <- molA] + [E <- compN]
//zero-coefficient E
//we create a molecule E at random location in the compartment to avoid
//rebinding effect, useful to maintain the concentration of a substrate species
//even after the reaction:
void DiffusionInfluencedReactionProcess::reactAtoC_compNtoE(Voxel* molA,
                                                            Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* mol(E->getRandomCompVoxel(1));
  if(mol)
    { 
      E->addMolecule(mol);
      C->addMolecule(molA, A->getTag(indexA));
      A->softRemoveMolecule(indexA);
      removeMolecule(B, molB, indexB);
    }
}

//A + B -> [C <- molN] + [B == D]
void DiffusionInfluencedReactionProcess::reactNtoC_BeqD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(C, molA, molB));
  if(mol)
    {
      C->addMolecule(mol, A->getTag(indexA));
      A->removeMolecule(indexA);
    }
}


//A + B -> [C <- molA] + [D <- molB]
void DiffusionInfluencedReactionProcess::reactAtoC_BtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molA, A->getTag(indexA));
  A->softRemoveMolecule(indexA);
  removeMolecule(B, molB, indexB, D);
}

//A + B -> [C <- molA] + [D <- molN]
void DiffusionInfluencedReactionProcess::reactAtoC_NtoD(
                                                  Voxel* molA, Voxel* molB,
                                                  const unsigned indexA,
                                                  const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(D, molA, molB));
  if(mol)
    {
      D->addMolecule(mol, B->getTag(indexB));
      C->addMolecule(molA, A->getTag(indexA));
      A->softRemoveMolecule(indexA);
      removeMolecule(B, molB, indexB);
    }
}

//A + B -> [C <- molB] + [D <- molA]
void DiffusionInfluencedReactionProcess::reactBtoC_AtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  C->addMolecule(molB, B->getTag(indexB));
  B->softRemoveMolecule(indexB);
  removeMolecule(A, molA, indexA, D);
}

//A + B -> [C <- molB] + [D <- molN]
void DiffusionInfluencedReactionProcess::reactBtoC_NtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(D, molA, molB));
  if(mol)
    {
      D->addMolecule(mol, A->getTag(indexA));
      C->addMolecule(molB, B->getTag(indexB));
      B->softRemoveMolecule(indexB);
      removeMolecule(A, molA, indexA);
    }
}

//A + B -> [C <- molN] + [D <- molN]
void DiffusionInfluencedReactionProcess::reactNtoC_NtoD(Voxel* molA,
                                                        Voxel* molB,
                                                        const unsigned indexA,
                                                        const unsigned indexB)
{
  Voxel* molC(getPopulatableVoxel(C, molA, molB));
  if(molC)
    {
      Voxel* molD(getPopulatableVoxel(C, molA, molB, molC));
      if(molD)
        {
          C->addMolecule(molC, A->getTag(indexA));
          D->addMolecule(molD, B->getTag(indexB));
          A->removeMolecule(indexA);
          removeMolecule(B, molB, indexB);
        }
    }
}

//A + B -> [A == C]
void DiffusionInfluencedReactionProcess::reactAeqC(Voxel* molA, Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  B->removeMolecule(indexB);
}

//A + B -> [B == C]
void DiffusionInfluencedReactionProcess::reactBeqC(Voxel* molA, Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  A->removeMolecule(indexA);
}

//A + B -> [C <- molA]
void DiffusionInfluencedReactionProcess::reactAtoC(Voxel* molA, Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  C->addMolecule(molA, A->getTag(indexA));
  A->softRemoveMolecule(indexA);
  removeMolecule(B, molB, indexB);
}

//A + B -> [C <- molB]
void DiffusionInfluencedReactionProcess::reactBtoC(Voxel* molA, Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  C->addMolecule(molB);
  B->softRemoveMolecule(indexB);
  removeMolecule(A, molA, indexA);
}

//A + B -> [C <- molB] + [tagC <- tagA]
void DiffusionInfluencedReactionProcess::reactBtoC_tagAtoC(Voxel* molA,
                                                           Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  C->addMolecule(molB, A->getTag(indexA));
  B->softRemoveMolecule(indexB);
  removeMolecule(A, molA, indexA);
}

//A + B -> [C <- molB] + [tagC <- tagB]
void DiffusionInfluencedReactionProcess::reactBtoC_tagBtoC(Voxel* molA,
                                                           Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  C->addMolecule(molB, B->getTag(indexB));
  B->softRemoveMolecule(indexB);
  removeMolecule(A, molA, indexA);
}

//A + B -> [C <- molN]
void DiffusionInfluencedReactionProcess::reactNtoC(Voxel* molA, Voxel* molB,
                                                   const unsigned indexA,
                                                   const unsigned indexB)
{
  Voxel* mol(getPopulatableVoxel(C, molA, molB));
  if(mol)
    {
      C->addMolecule(mol, A->getTag(indexA));
      A->removeMolecule(indexA);
      removeMolecule(B, molB, indexB);
    }
}

void DiffusionInfluencedReactionProcess::setReactMethod()
{
  if(variableC && D)
    {
      if(A == D)
        {
          //A + B -> variableC + [A == D]
          reactM = &DiffusionInfluencedReactionProcess::reactVarC_AeqD;
        }
      else if(B == D)
        {
          //A + B -> variableC + [B == D]
          reactM = &DiffusionInfluencedReactionProcess::reactVarC_BeqD;
        }
      else
        { 
          if(A->isReplaceable(D))
            {
              //A + B -> variableC + [D <- molA]
              reactM = &DiffusionInfluencedReactionProcess::reactVarC_AtoD;
            }
          else if(B->isReplaceable(D))
            {
              //A + B -> variableC + [D <- molB]
              reactM = &DiffusionInfluencedReactionProcess::reactVarC_BtoD;
            }
          else
            {
              //A + B -> variableC + [D <- molN]
              reactM = &DiffusionInfluencedReactionProcess::reactVarC_NtoD;
            }
        }
    }
  else if(variableD && C)
    {
      if(A == C)
        {
          //A + B -> variableD + [A == C]
          reactM = &DiffusionInfluencedReactionProcess::reactVarD_AeqC;
        }
      else if(B == C)
        {
          //A + B -> variableD + [B == C]
          reactM = &DiffusionInfluencedReactionProcess::reactVarD_BeqC;
        }
      else
        { 
          if(A->isReplaceable(C))
            {
              //A + B -> variableD + [C <- molA]
              reactM = &DiffusionInfluencedReactionProcess::reactVarD_AtoC;
            }
          else if(B->isReplaceable(C))
            {
              //A + B -> variableD + [C <- molB]
              reactM = &DiffusionInfluencedReactionProcess::reactVarD_BtoC;
            }
          else
            {
              //A + B -> variableD + [C <- molN]
              reactM = &DiffusionInfluencedReactionProcess::reactVarD_NtoC;
            }
        }
    }
  else if(variableC)
    {
      if(variableD)
        {
          //A + B -> variableC + variableD
          reactM = &DiffusionInfluencedReactionProcess::reactVarC_VarD;
        }
      else
        {
          //A + B -> variableC
          reactM = &DiffusionInfluencedReactionProcess::reactVarC;
        }
    }
  else if(D)
    {
      if(A == C && B == D)
        {
          //A + B -> [A == C] + [B == D]
          reactM = &DiffusionInfluencedReactionProcess::reactNone;
        }
      else if(B == C && A == D)
        {
          //A + B -> [B == C] + [A == D]
          reactM = &DiffusionInfluencedReactionProcess::reactNone;
        }
      else if(A == C)
        {
          if(B->isReplaceable(D))
            {
              //A + B -> [A == C] + [D <- molB]
              reactM = &DiffusionInfluencedReactionProcess::reactAeqC_BtoD;
            }
          else
            {
              //A + B -> [A == C] + [D <- molN]
              reactM = &DiffusionInfluencedReactionProcess::reactAeqC_NtoD;
            }
        }
      else if(B == C)
        {
          if(A->isReplaceable(D))
            {
              //A + B -> [B == C] + [D <- molA]
              reactM = &DiffusionInfluencedReactionProcess::reactBeqC_AtoD;
            }
          else
            {
              //A + B -> [B == C] + [D <- molN]
              reactM = &DiffusionInfluencedReactionProcess::reactBeqC_NtoD;
            }
        }
      else if(A == D)
        {
          if(B->isReplaceable(C))
            {
              //A + B -> [C <- molB] + [A == D]
              reactM = &DiffusionInfluencedReactionProcess::reactBtoC_AeqD;
            }
          else
            {
              //A + B -> [C <- molN] + [A == D]
              reactM = &DiffusionInfluencedReactionProcess::reactNtoC_AeqD;
            }
        }
      else if(B == D)
        {
          if(A->isReplaceable(C))
            {
              if(C->getIsTagged() && A->getIsTagged())
                {
                  //A + B -> [C <- molA] + [tagC <- tagA] + [B == D]
                  reactM = 
                    &DiffusionInfluencedReactionProcess::reactAtoC_BeqD_tagAtoC;
                }
              else
                {
                  //A + B -> [C <- molA] + [B == D]
                  reactM = &DiffusionInfluencedReactionProcess::reactAtoC_BeqD;
                }
            }
          else
            {
              //A + B -> [C <- molN] + [B == D]
              reactM = &DiffusionInfluencedReactionProcess::reactNtoC_BeqD;
            }
        }
      else
        {
          if(A->isReplaceable(C))
            {
              if(B->isReplaceable(D))
                {
                  //A + B -> [C <- molA] + [D <- molB]
                  reactM = &DiffusionInfluencedReactionProcess::reactAtoC_BtoD;
                }
              else
                {
                  //A + B -> [C <- molA] + [D <- molN]
                  reactM = &DiffusionInfluencedReactionProcess::reactAtoC_NtoD;
                }
            }
          else if(B->isReplaceable(C))
            {
              if(A->isReplaceable(D))
                {
                  //A + B -> [C <- molB] + [D <- molA]
                  reactM = &DiffusionInfluencedReactionProcess::reactBtoC_AtoD;
                }
              else
                {
                  //A + B -> [C <- molB] + [D <- molN]
                  reactM = &DiffusionInfluencedReactionProcess::reactBtoC_NtoD;
                }
            }
          else
            {
              //A + B -> [C <- molN] + [D <- molN]
              reactM = &DiffusionInfluencedReactionProcess::reactNtoC_NtoD;
            }
        }
    }
  //A + B -> C + E(0 coefficient, random comp voxel)
  else if(E)
    {
      if(A == C)
        {
          throwException("reactAeqC_E");
        }
      else if(B == C)
        {
          throwException("reactBeqC_E");
        }
      else if(A->isReplaceable(C))
        {
          reactM = &DiffusionInfluencedReactionProcess::reactAtoC_compNtoE;
        }
      else
        {
          throwException("reactBtoC_E");
        }
    }
  else
    {
      if(A == C)
        {
          //A + B -> [A == C]
          reactM = &DiffusionInfluencedReactionProcess::reactAeqC;
        }
      else if(B == C)
        {
          //A + B -> [B == C]
          reactM = &DiffusionInfluencedReactionProcess::reactBeqC;
        }
      else if(A->isReplaceable(C))
        {
          //A + B -> [C <- molA]
          reactM = &DiffusionInfluencedReactionProcess::reactAtoC;
        }
      else if(B->isReplaceable(C))
        {
          if(C->getIsTagged())
            {
              if(B->getIsTagged())
                {
                  //A + B -> [C <- molB] + [tagC <- tagB]
                  reactM = 
                    &DiffusionInfluencedReactionProcess::reactBtoC_tagBtoC;
                }
              else
                {
                  //A + B -> [C <- molB] + [tagC <- tagA]
                  reactM =
                    &DiffusionInfluencedReactionProcess::reactBtoC_tagAtoC;
                }
            }
          else
            {
              //A + B -> [C <- molB]
              reactM = &DiffusionInfluencedReactionProcess::reactBtoC;
            }
        }
      else
        {
          //A + B -> [C <- molN]
          reactM = &DiffusionInfluencedReactionProcess::reactNtoC;
        }
    }
}

void DiffusionInfluencedReactionProcess::throwException(String aString)
{
  THROW_EXCEPTION(ValueError, String(getPropertyInterface().getClassName()) +
                  "[" + getFullID().asString() + "]: " + aString + " is not " +
                  "yet implemented.");
}

//positive-coefficient F
void DiffusionInfluencedReactionProcess::addMoleculeF()
{
  if(!F)
    {
      return;
    }
  moleculeF = F->getRandomAdjoiningVoxel(moleculeC, SearchVacant);
  if(moleculeF == NULL)
    {
      moleculeF = F->getRandomAdjoiningVoxel(moleculeD, SearchVacant);
      if(moleculeF == NULL)
        {
          return;
        }
    }
  F->addMolecule(moleculeF);
}

void DiffusionInfluencedReactionProcess::calculateReactionProbability()
{
  //Refer to the paper for the description of the variables used in this
  //method.
  if(A->getDimension() == 3 && B->getDimension() == 3)
    {
      if(A != B)
        {
          if(p == -1)
            {
              p = k/(6*sqrt(2)*(D_A+D_B)*r_v);
            }
          else
            {
              k = p*(6*sqrt(2)*(D_A+D_B)*r_v);
            }
        }
      else
        {
          if(p == -1)
            {
              p = k/(6*sqrt(2)*D_A*r_v);
            }
          else
            {
              k = p*(6*sqrt(2)*D_A*r_v);
            }
        }
    }
  else if(A->getDimension() != 3 && B->getDimension() != 3)
    {
      //Inter-surface Comp reaction.
      //For surface edge absorbing reactions:
      if(A->getComp() != B->getComp())
        {
          k = p;
        }
      else if(A != B)
        {
          if(p == -1)
            {
              p = pow(2*sqrt(2)+4*sqrt(3)+3*sqrt(6)+sqrt(22), 2)*k/
                (72*(6*sqrt(2)+4*sqrt(3)+3*sqrt(6))*(D_A+D_B));
            }
          else
            {
              k = p*(72*(6*sqrt(2)+4*sqrt(3)+3*sqrt(6))*(D_A+D_B))/
                pow(2*sqrt(2)+4*sqrt(3)+3*sqrt(6)+sqrt(22), 2);
            }
        }
      else
        {
          if(p == -1)
            {
              p = pow(2*sqrt(2)+4*sqrt(3)+3*sqrt(6)+sqrt(22), 2)*k/
                (72*(6*sqrt(2)+4*sqrt(3)+3*sqrt(6))*(D_A));
            }
          else
            {
              k = p*(72*(6*sqrt(2)+4*sqrt(3)+3*sqrt(6))*(D_A))/
                pow(2*sqrt(2)+4*sqrt(3)+3*sqrt(6)+sqrt(22), 2);
            }
        }
    }
  else if(A->getDimension() == 3 && B->getIsLipid())
    {
      if(p == -1)
        {
          p = 24*k*r_v/((6+3*sqrt(3)+2*sqrt(6))*D_A);
        }
      else
        {
          k = p*((6+3*sqrt(3)+2*sqrt(6))*D_A)/(24*r_v);
        }
    }
  else if(A->getIsLipid() && B->getDimension() == 3)
    {
      if(p == -1)
        {
          p = 24*k*r_v/((6+3*sqrt(3)+2*sqrt(6))*D_B);
        }
      else
        {
          k = p*((6+3*sqrt(3)+2*sqrt(6))*D_B)/(24*r_v);
        }
    }
  else if(A->getDimension() == 3 && B->getDimension() != 3)
    {
      if(p == -1)
        {
          p = sqrt(2)*k/(3*D_A*r_v);
        }
      else
        {
          k = p*(3*D_A*r_v)/sqrt(2);
        }
    }
  else if(A->getDimension() != 3 && B->getDimension() == 3)
    {
      if(p == -1)
        {
          p = sqrt(2)*k/(3*D_B*r_v);
        }
      else
        {
          k = p*(3*D_B*r_v)/sqrt(2);
        }
    }
  else
    {
      THROW_EXCEPTION(ValueError, 
                      String(getPropertyInterface().getClassName()) + 
                      " [" + getFullID().asString() + 
                      "]: Error in type of second order reaction.");
    }
}

void DiffusionInfluencedReactionProcess::printParameters()
{
  String aProcess(String(getPropertyInterface().getClassName()) + 
                                      "[" + getFullID().asString() + "]");
  cout << aProcess << std::endl;
  cout << "  " << getIDString(A) << " + " <<  getIDString(B) << " -> ";
  if(C)
    {
      cout << getIDString(C);
    }
  else
    {
      cout << getIDString(variableC);
    }
  if(D)
    {
      cout << " + " << getIDString(D);
    }
  else if(variableD)
    {
      cout << " + " << getIDString(variableD);
    }
  cout << ": k=" << k << ", p=" << p << 
    ", p_A=" << A->getReactionProbability(B->getID()) <<
    ", p_B=" << B->getReactionProbability(A->getID()) << std::endl; 
}

