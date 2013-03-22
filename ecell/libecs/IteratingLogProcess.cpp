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

#include "IteratingLogProcess.hpp"

void IteratingLogProcess::initializeFifth()
{
  for(std::vector<Species*>::const_iterator i(theProcessSpecies.begin());
      i != theProcessSpecies.end(); ++i)
    {
      if((*i)->getDiffusionInterval() < theStepInterval)
        {
          theStepInterval = (*i)->getDiffusionInterval();
        }
      Species* reactantPair((*i)->getDiffusionInfluencedReactantPair());
      if(reactantPair != NULL && 
         reactantPair->getDiffusionInterval() < theStepInterval)
        {
          theStepInterval = reactantPair->getDiffusionInterval();
        }
    }
  if(LogInterval > 0)
    {
      theStepInterval = LogInterval;
    }
  else
    {
      LogInterval = theStepInterval;
    }
  theTime = LogStart;
  thePriorityQueue->move(theQueueID);
}

void IteratingLogProcess::initializeLastOnce()
{
  theLogFile.open(FileName.c_str(), std::ios::trunc);
  theTotalIterations = Iterations;
  if(RebindTime)
    {
      timePoints = Iterations;
    }
  else
    {
      timePoints = (unsigned int)ceil((LogEnd-LogStart)/theStepInterval)+1;
    }
  theLogValues.resize(timePoints);
  for(unsigned int i(0); i != timePoints; ++i)
    {
      theLogValues[i].resize(theProcessSpecies.size()+
                             theProcessVariables.size());
      for(unsigned int j(0);
          j != theProcessSpecies.size()+theProcessVariables.size(); ++j)
        {
          theLogValues[i][j] = 0;
        }
    }
}

void IteratingLogProcess::fire()
{
  if(theTime >= LogStart && theTime <= LogEnd)
    {
      logValues();
      //If all survival species are dead, go on to the next iteration:
      if((Survival || RebindTime) && !isSurviving)
        {
          theStepInterval = libecs::INF;
        }
      ++timePointCnt;
    }
  if(theTime >= LogEnd && Iterations > 0)
    {
      theStepInterval = LogInterval;
      --Iterations;
      std::cout << "Iterations left:" << Iterations << " of " <<
        theTotalIterations << std::endl;
      if(Iterations > 0)
        {
          theSpatiocyteStepper->reset(Iterations);
          saveBackup();
          return;
        }
    }
  if(Iterations == 0)
    {
      saveFile();
      std::cout << "Done saving." << std::endl;
    }
  theTime += theStepInterval;
  thePriorityQueue->moveTop();
}


void IteratingLogProcess::saveFile()
{
  std::cout << "Saving data in: " << FileName.c_str() << std::endl;
  double aTime(LogInterval);
  for(unsigned int i(0); i != timePoints; ++i)
    {
      theLogFile << std::setprecision(15) << aTime;
      for(unsigned int j(0);
          j != theProcessSpecies.size()+theProcessVariables.size(); ++j)
        {
          theLogFile << "," << std::setprecision(15) <<
            theLogValues[i][j]/theTotalIterations;
        }
      theLogFile << std::endl;
      aTime += LogInterval;
    }
  theLogFile.close();
  theStepInterval = libecs::INF;
}

void IteratingLogProcess::saveBackup()
{
  if(SaveCounts > 0 && 
     Iterations%(int)rint(theTotalIterations/SaveCounts) == 0)
    {
      std::string aFileName(FileName.c_str());
      aFileName = aFileName + ".back";
      std::cout << "Saving backup data in: " << aFileName << std::endl;
      std::ofstream aFile;
      aFile.open(aFileName.c_str(), std::ios::trunc);
      double aTime(LogInterval);
      int completedIterations(theTotalIterations-Iterations);
      for(unsigned int i(0); i != timePoints; ++i)
        {
          aFile << std::setprecision(15) << aTime;
          for(unsigned int j(0);
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
 //std::cout << "timePoint:" << timePointCnt <<  " curr:" << theSpatiocyteStepper->getCurrentTime() << std::endl;
  isSurviving = false;
  for(unsigned int i(0); i != theProcessSpecies.size(); ++i)
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
      else if(Displacement)
        {
          theLogValues[timePointCnt][i] += 
            aSpecies->getMeanSquaredDisplacement();
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
  for(unsigned int i(0); i != theProcessVariables.size(); ++i)
    {
      theLogValues[timePointCnt][i+size] += theProcessVariables[i]->getValue();
    }
}


LIBECS_DM_INIT(IteratingLogProcess, Process); 
