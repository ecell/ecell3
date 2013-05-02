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

#include <LifetimeLogProcess.hpp>

void LifetimeLogProcess::initialize()
{
  if(isInitialized)
    {
      return;
    }
  SpatiocyteProcess::initialize();
  isPriorityQueued = true;
  theTotalIterations = Iterations;
}

void LifetimeLogProcess::initializeFirst()
{
  SpatiocyteProcess::initializeFirst();
  isBindingSite.resize(getStepper()->getProcessVector().size(), false);
  isTrackedSpecies.resize(theSpecies.size(), false);
  isUntrackedSpecies.resize(theSpecies.size(), false);
  unsigned cntTracked(0);
  unsigned cntUntracked(0);
  for(VariableReferenceVector::iterator
      i(theVariableReferenceVector.begin());
      i != theVariableReferenceVector.end(); ++i)
    {
      Species* aSpecies(theSpatiocyteStepper->variable2species(
                                     (*i).getVariable())); 
      if((*i).getCoefficient() == -1)
        {
          isTrackedSpecies[aSpecies->getID()] = true;
          aSpecies->setIsTagged();
          ++cntTracked;
        }
      else if((*i).getCoefficient() == 1)
        {
          isUntrackedSpecies[aSpecies->getID()] = true;
          ++cntUntracked;
        }
    }
  if(!cntTracked)
    {
      THROW_EXCEPTION(ValueError, String(
                      getPropertyInterface().getClassName()) +
                      "[" + getFullID().asString() + 
                      "]: Lifetime logging requires at least one " +
                      "nonHD variable reference with -1 " +
                      "coefficient as the tracked species, " +
                      "but none is given."); 
    }
  if(!cntUntracked)
    {
      THROW_EXCEPTION(ValueError, String(
                      getPropertyInterface().getClassName()) +
                      "[" + getFullID().asString() + 
                      "]: Lifetime logging requires at least one " +
                      "nonHD variable reference with 1 " +
                      "coefficient as the untracked species, " +
                      "but none is given."); 
    }
}

void LifetimeLogProcess::initializeSecond()
{
  availableTagIDs.resize(0);
  theTagTimes.resize(0);
}

void LifetimeLogProcess::initializeLastOnce()
{
  theLogFile.open(FileName.c_str(), std::ios::trunc);
}

void LifetimeLogProcess::initializeFifth()
{
  theInterval = LogEnd;
  if(!LogStart)
    {
      LogStart = theInterval;
    }
  theTime = std::max(LogStart-theInterval, getStepper()->getMinStepInterval());
  thePriorityQueue->move(theQueueID);
}

void LifetimeLogProcess::fire()
{
  if(theTime < LogStart)
    {
      cout << "Iterations left:" << Iterations << " of " <<
        theTotalIterations << std::endl;    
    }
  if(theTime >= LogEnd && Iterations > 0)
    {
      --Iterations;
      if(Iterations)
        {
          theSpatiocyteStepper->reset(Iterations);
          return;
        }
      else
        {
          theInterval = libecs::INF;
        }
    }
  theTime += theInterval;
  thePriorityQueue->moveTop();
}

void LifetimeLogProcess::interruptedPre(ReactionProcess* aProcess)
{
  if(aProcess->getA() && isTrackedSpecies[aProcess->getA()->getID()])
    {
      logTrackedMolecule(aProcess, aProcess->getA(), aProcess->getMoleculeA());
    }
  else if(aProcess->getB() && isTrackedSpecies[aProcess->getB()->getID()])
    {
      logTrackedMolecule(aProcess, aProcess->getB(), aProcess->getMoleculeB());
    }
}

void LifetimeLogProcess::interruptedPost(ReactionProcess* aProcess)
{
  if(aProcess->getC() && isTrackedSpecies[aProcess->getC()->getID()])
    {
      initTrackedMolecule(aProcess->getC());
    }
  else if(aProcess->getD() && isTrackedSpecies[aProcess->getD()->getID()])
    {
      initTrackedMolecule(aProcess->getD());
    }
}

void LifetimeLogProcess::logTrackedMolecule(ReactionProcess* aProcess,
                                            Species* aSpecies,
                                            const Voxel* aMolecule)
{
  if(isBindingSite[aProcess->getID()])
    {
      if(aProcess->getMoleculeC() &&
         isTrackedSpecies[aProcess->getC()->getID()])
        {
          return;
        }
      else if(aProcess->getMoleculeD() &&
              isTrackedSpecies[aProcess->getD()->getID()])
        {
          return;
        }
    }
  const unsigned anIndex(aSpecies->getIndex(aMolecule));
  const Point aPoint(aSpecies->getPoint(anIndex));
  Tag& aTag(aSpecies->getTag(anIndex));
  const Point anOrigin(aSpecies->coord2point(aTag.origin));
  availableTagIDs.push_back(aTag.id);
  double aTime(getStepper()->getCurrentTime());
  theLogFile << std::setprecision(15) << aTime-theTagTimes[aTag.id] << "," <<
    distance(aPoint, anOrigin)*2*theSpatiocyteStepper->getVoxelRadius() << ","
    << theTagTimes[aTag.id] << "," << aTime << "," << konCnt << std::endl;
}

void LifetimeLogProcess::initTrackedMolecule(Species* aSpecies)
{
  const unsigned anIndex(aSpecies->size()-1);
  Tag& aTag(aSpecies->getTag(anIndex));
  aTag.origin = aSpecies->getCoord(anIndex);
  if(availableTagIDs.size())
    {
      aTag.id = availableTagIDs.back();
      theTagTimes[availableTagIDs.back()] = getStepper()->getCurrentTime();
      availableTagIDs.pop_back();
    }
  else
    {
      aTag.id = theTagTimes.size();
      theTagTimes.push_back(getStepper()->getCurrentTime());
    }
  ++konCnt;
}

bool LifetimeLogProcess::isDependentOnPre(const ReactionProcess* aProcess)
{
  const VariableReferenceVector& aVariableReferences(
                                       aProcess->getVariableReferenceVector()); 
  for(unsigned i(0); i != isTrackedSpecies.size(); ++i)
    {
      if(isTrackedSpecies[i] && isInVariableReferences(
           aVariableReferences, -1, theSpecies[i]->getVariable()))
        {
          for(unsigned j(0); j != isUntrackedSpecies.size(); ++j)
            {
              if(isUntrackedSpecies[j] && isInVariableReferences(
                 aVariableReferences, 1, theSpecies[j]->getVariable()))
                {
                  for(unsigned k(0); k != isTrackedSpecies.size(); ++k)
                    {
                      if(isTrackedSpecies[k] && isInVariableReferences(
                         aVariableReferences, 1, theSpecies[k]->getVariable())) 
                        {
                          isBindingSite[aProcess->getID()] = true;
                          return true;
                        }
                    }
                  return true;
                }
            }
        }
    }
  return false;
}

bool LifetimeLogProcess::isInVariableReferences(const VariableReferenceVector&
                                                aVariableReferences,
                                                const int aCoefficient,
                                                const Variable* aVariable) const
{
  for(VariableReferenceVector::const_iterator
      i(aVariableReferences.begin()); i != aVariableReferences.end(); ++i)
    {
      //If the both coefficients have the same sign:
      if((*i).getCoefficient()*aCoefficient > 0 &&
         (*i).getVariable() == aVariable)
        {
          return true;
        }
    }
  return false;
}

bool LifetimeLogProcess::isDependentOnPost(const ReactionProcess* aProcess)
{
  const VariableReferenceVector& aVariableReferences(
                                       aProcess->getVariableReferenceVector()); 
  for(unsigned i(0); i != isTrackedSpecies.size(); ++i)
    {
      if(isTrackedSpecies[i] && isInVariableReferences(
           aVariableReferences, 1, theSpecies[i]->getVariable()))
        {
          for(unsigned j(0); j != isTrackedSpecies.size(); ++j)
            {
              if(isTrackedSpecies[j] && isInVariableReferences(
                   aVariableReferences, -1, theSpecies[j]->getVariable()))
                {
                  return false;
                }
            }
          return true;
        }
    }
  return false;
}


LIBECS_DM_INIT(LifetimeLogProcess, Process); 
