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

#include <IteratingLogProcess.hpp>

void IteratingLogProcess::initializeFifth()
{
  for(std::vector<Species*>::const_iterator i(theProcessSpecies.begin());
      i != theProcessSpecies.end(); ++i)
    {
      if((*i)->getDiffusionInterval() < theInterval)
        {
          theInterval = (*i)->getDiffusionInterval();
        }
      Species* reactantPair((*i)->getDiffusionInfluencedReactantPair());
      if(reactantPair != NULL && 
         reactantPair->getDiffusionInterval() < theInterval)
        {
          theInterval = reactantPair->getDiffusionInterval();
        }
    }
  if(LogInterval > 0)
    {
      theInterval = LogInterval;
    }
  else
    {
      LogInterval = theInterval;
    }
  if(!LogStart)
    {
      LogStart = theInterval;
    }
  theTime = std::max(LogStart-theInterval, getStepper()->getMinStepInterval());
  thePriorityQueue->move(theQueueID);
  if(theFileCnt-FileStartCount)
    {
      initializeLastOnce();
    }
}

void IteratingLogProcess::initializeLastOnce()
{
  if(SeparateFiles)
    {
      SaveCounts = 0;
      ++theFileCnt;
    }
  if(RebindTime)
    {
      timePoints = Iterations;
    }
  else
    {
      timePoints = (unsigned)ceil((LogEnd-LogStart)/theInterval)+1;
    }
  theLogValues.resize(timePoints);
  unsigned aDataSize(theProcessSpecies.size()+theProcessVariables.size());
  if(FrameDisplacement)
    {
      aDataSize = 0;
      for(unsigned i(0); i != theProcessSpecies.size(); ++i)
        {
          aDataSize += theProcessSpecies[i]->size();
        }
    }
  for(unsigned i(0); i != timePoints; ++i)
    {
      theLogValues[i].resize(aDataSize, 0);
      for(unsigned j(0); j != aDataSize; ++j)
        {
          theLogValues[i][j] = 0;
        }
    }
}

void IteratingLogProcess::fire()
{
  if(theTime < LogStart)
    {
      doPreLog();
      theTime = LogStart;
      thePriorityQueue->moveTop();
      return;
    }
  else if(theTime >= LogStart && theTime <= LogEnd)
    {
      logValues();
      //If all survival species are dead, go on to the next iteration:
      if((Survival || RebindTime) && !isSurviving)
        {
          theInterval = libecs::INF;
        }
      ++timePointCnt;
    }
  if(theTime >= LogEnd && Iterations > 0)
    {
      --Iterations;
      saveFile();
      if(Iterations)
        {
          saveBackup();
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


void IteratingLogProcess::doPreLog()
{
  cout << "Iterations left:" << Iterations << " of " << theTotalIterations
    << std::endl;
  for(unsigned i(0); i != theProcessSpecies.size(); ++i)
    {
      Species* aSpecies(theProcessSpecies[i]);
      if(FrameDisplacement)
        {
          aSpecies->resetMoleculeOrigins();
        }
    }
}

void IteratingLogProcess::saveFile()
{
  unsigned aCompletedIterations(1);
  if(SeparateFiles)
    {
      String aFileName(FileName+int2str(theFileCnt-1));
      cout << "Saving data in: " << aFileName.c_str() << std::endl;
      theLogFile.open(aFileName.c_str(), std::ios::trunc);
    }
  else if(!Iterations)
    {
      cout << "Saving data in: " << FileName.c_str() << std::endl;
      theLogFile.open(FileName.c_str(), std::ios::trunc);
      aCompletedIterations = theTotalIterations;
    }
  else
    {
      return;
    }
  double aTime(LogStart);
  for(unsigned i(0); i < timePoints-1; ++i)
    {
      theLogFile << std::setprecision(15) << aTime;
      for(unsigned j(0); j != theLogValues[i].size(); ++j)
        {
          theLogFile << "," << std::setprecision(15) <<
            theLogValues[i][j]/aCompletedIterations;
        }
      theLogFile << std::endl;
      aTime += LogInterval;
    }
  theLogFile.close();
}

void IteratingLogProcess::saveBackup()
{
  if(SaveCounts > 0 && 
     Iterations%(unsigned)rint(theTotalIterations/SaveCounts) == 0)
    {
      std::string aFileName(FileName.c_str());
      aFileName = aFileName + ".back";
      cout << "Saving backup data in: " << aFileName << std::endl;
      std::ofstream aFile;
      aFile.open(aFileName.c_str(), std::ios::trunc);
      double aTime(LogStart);
      unsigned completedIterations(theTotalIterations-Iterations);
      for(unsigned i(0); i < timePoints-2; ++i)
        {
          aFile << std::setprecision(15) << aTime;
          for(unsigned j(0);
              j != theProcessSpecies.size()+theProcessVariables.size(); ++j)
            {
              aFile << "," << std::setprecision(15) <<
                theLogValues[i][j]/completedIterations;
            }
          aFile << std::endl;
          aTime += LogInterval;
        }
      aFile.close();
    }
}

void IteratingLogProcess::logValues()
{
 //cout << "timePoint:" << timePointCnt <<  " curr:" << theSpatiocyteStepper->getCurrentTime() << std::endl;
  isSurviving = false;
  for(unsigned i(0); i != theProcessSpecies.size(); ++i)
    {
      Species* aSpecies(theProcessSpecies[i]);
      if(RebindTime)
        {
          if(aSpecies->getVariable()->getValue())
            {
              isSurviving = true;
            }
          if(!theLogValues[theTotalIterations-Iterations][i])
            {
              if(!aSpecies->getVariable()->getValue())
                {
                  theLogValues[theTotalIterations-Iterations][i] = theTime;
                }
            }
        }
      else if(Survival)
        {
          if(aSpecies->getVariable()->getValue())
            {
              isSurviving = true;
            }
          theLogValues[timePointCnt][i] += aSpecies->getVariable()->getValue()/
            aSpecies->getInitCoordSize();
        }
      else if(SquaredDisplacement)
        {
          theLogValues[timePointCnt][i] += 
            aSpecies->getMeanSquaredDisplacement();
        }
      else if(FrameDisplacement)
        {
          for(unsigned j(0); j != aSpecies->size(); ++j)
            {
              theLogValues[timePointCnt][i*aSpecies->size()+j] += 
                sqrt(aSpecies->getSquaredDisplacement(j)); 
            }
          aSpecies->resetMoleculeOrigins();
        }
      else if(Diffusion)
        {
          double aCoeff(aSpecies->getDimension()*2);
          theLogValues[timePointCnt][i] += 
            aSpecies->getMeanSquaredDisplacement()/(aCoeff*theTime);
        }
      //By default log the values:
      else
        {
          theLogValues[timePointCnt][i] += aSpecies->getVariable()->getValue();
        }
    }
  unsigned size(theProcessSpecies.size());
  for(unsigned i(0); i != theProcessVariables.size(); ++i)
    {
      theLogValues[timePointCnt][i+size] += theProcessVariables[i]->getValue();
    }
}


LIBECS_DM_INIT(IteratingLogProcess, Process); 
