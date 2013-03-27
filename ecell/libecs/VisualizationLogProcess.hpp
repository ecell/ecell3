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


#ifndef __VisualizationLogProcess_hpp
#define __VisualizationLogProcess_hpp

#include <fstream> //provides ofstream
#include <libecs/MethodProxy.hpp>
#include "SpatiocyteProcess.hpp"
#include "SpatiocyteSpecies.hpp"

namespace libecs
{

LIBECS_DM_CLASS(VisualizationLogProcess, SpatiocyteProcess)
{ 
public:
  LIBECS_DM_OBJECT(VisualizationLogProcess, Process)
    {
      INHERIT_PROPERTIES(Process);
      PROPERTYSLOT_SET_GET(Integer, Polymer);
      PROPERTYSLOT_SET_GET(Real, LogInterval);
      PROPERTYSLOT_SET_GET(String, FileName);
    }
  VisualizationLogProcess():
    Polymer(1),
    theLogMarker(UINT_MAX),
    theMeanCount(0),
    LogInterval(0),
    FileName("VisualLog.dat") {}
  virtual ~VisualizationLogProcess() {}
  SIMPLE_SET_GET_METHOD(Integer, Polymer);
  SIMPLE_SET_GET_METHOD(Real, LogInterval);
  SIMPLE_SET_GET_METHOD(String, FileName); 
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
          Variable* aVariable((*i).getVariable());
          if(aVariable->getName() == "HD")
            {
              THROW_EXCEPTION(ValueError, 
                getPropertyInterface().getClassName() +
                " [" + getFullID().asString() + "]: " +  
                aVariable->getFullID().asString() + " is a HD species and " +
                "therefore cannot be visualized.");
            }
        }
    }	
  virtual void initializeFourth()
    {
      for(unsigned int i(0); i != theProcessSpecies.size(); ++i)
        {
          Species* aSpecies(theProcessSpecies[i]);
          if(aSpecies->getIsOffLattice())
            {
              theOffLatticeSpecies.push_back(aSpecies);
            }
          else
            {
              theLatticeSpecies.push_back(aSpecies);
              if(aSpecies->getIsPolymer() && Polymer)
                {
                  thePolymerSpecies.push_back(aSpecies);
                  thePolymerIndex.push_back(theLatticeSpecies.size()-1);
                }
            }

        }
      thePriority = -1;
    }
  virtual void initializeFifth()
    {
      if(LogInterval > 0)
        {
          theStepInterval = LogInterval;
        }
      else
        {
          //Use the smallest time step of all queued events for
          //the step interval:
          theTime = libecs::INF;
          thePriorityQueue->move(theQueueID);
          theStepInterval = thePriorityQueue->getTop()->getTime();
        }
      theTime = theStepInterval;
      thePriorityQueue->move(theQueueID);
    }
  virtual void initializeLastOnce()
    {
      std::ostringstream aFilename;
      aFilename << FileName << std::ends;
      theLogFile.open(aFilename.str().c_str(), std::ios::binary |
                      std::ios::trunc);
      initializeLog();
      logCompVacant();
      logSpecies();
      theLogFile.flush();
    }
  virtual void fire()
    {
      logSpecies();
      if(LogInterval > 0)
        {
          theTime += LogInterval;
          thePriorityQueue->moveTop();
        }
      else
        {
          //get the next step interval of the SpatiocyteStepper:
          double aTime(theTime);
          theTime = libecs::INF;
          thePriorityQueue->moveTop();
          if(thePriorityQueue->getTop()->getTime() > aTime)
            {
              theStepInterval = thePriorityQueue->getTop()->getTime() -
                theSpatiocyteStepper->getCurrentTime();
            }
          theTime = aTime + theStepInterval;
          thePriorityQueue->move(theQueueID);
        }
    }
protected:
  virtual void initializeLog();
  virtual void logCompVacant();
  void logSpecies();
  void logMolecules(int);
  void logSourceMolecules(int);
  void logTargetMolecules(int);
  void logSharedMolecules(int);
  void logPolymers(int);
  void logOffLattice(int);
protected:
  unsigned int Polymer;
  unsigned int theLogMarker;
  unsigned int theMeanCount;
  double LogInterval;
  String FileName;
  std::ofstream theLogFile;
  std::streampos theStepStartPos;  
  std::vector<unsigned int> thePolymerIndex;
  std::vector<Species*> thePolymerSpecies;
  std::vector<Species*> theLatticeSpecies;
  std::vector<Species*> theOffLatticeSpecies;
};

}

#endif /* __VisualizationLogProcess_hpp */
