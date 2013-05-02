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


#ifndef __IteratingLogProcess_hpp
#define __IteratingLogProcess_hpp

#include <fstream>
#include <math.h>
#include <SpatiocyteProcess.hpp>
#include <ReactionProcess.hpp>
#include <SpatiocyteSpecies.hpp>

LIBECS_DM_CLASS(IteratingLogProcess, SpatiocyteProcess)
{ 
public:
  LIBECS_DM_OBJECT(IteratingLogProcess, Process)
    {
      INHERIT_PROPERTIES(Process);
      PROPERTYSLOT_SET_GET(Integer, Centered);
      PROPERTYSLOT_SET_GET(Integer, Diffusion);
      PROPERTYSLOT_SET_GET(Integer, FileStartCount);
      PROPERTYSLOT_SET_GET(Integer, FrameDisplacement);
      PROPERTYSLOT_SET_GET(Integer, InContact);
      PROPERTYSLOT_SET_GET(Integer, Iterations);
      PROPERTYSLOT_SET_GET(Integer, RebindTime);
      PROPERTYSLOT_SET_GET(Integer, SaveCounts);
      PROPERTYSLOT_SET_GET(Integer, SeparateFiles);
      PROPERTYSLOT_SET_GET(Integer, SquaredDisplacement);
      PROPERTYSLOT_SET_GET(Integer, Survival);
      PROPERTYSLOT_SET_GET(Real, LogEnd);
      PROPERTYSLOT_SET_GET(Real, LogInterval);
      PROPERTYSLOT_SET_GET(Real, LogStart);
      PROPERTYSLOT_SET_GET(String, FileName);
    }
  SIMPLE_SET_GET_METHOD(Integer, Centered);
  SIMPLE_SET_GET_METHOD(Integer, Diffusion);
  SIMPLE_SET_GET_METHOD(Integer, FileStartCount);
  SIMPLE_SET_GET_METHOD(Integer, FrameDisplacement);
  SIMPLE_SET_GET_METHOD(Integer, InContact);
  SIMPLE_SET_GET_METHOD(Integer, Iterations);
  SIMPLE_SET_GET_METHOD(Integer, RebindTime);
  SIMPLE_SET_GET_METHOD(Integer, SaveCounts);
  SIMPLE_SET_GET_METHOD(Integer, SeparateFiles);
  SIMPLE_SET_GET_METHOD(Integer, SquaredDisplacement);
  SIMPLE_SET_GET_METHOD(Integer, Survival);
  SIMPLE_SET_GET_METHOD(Real, LogEnd);
  SIMPLE_SET_GET_METHOD(Real, LogStart);
  SIMPLE_SET_GET_METHOD(Real, LogInterval);
  SIMPLE_SET_GET_METHOD(String, FileName);
  IteratingLogProcess():
    SpatiocyteProcess(),
    Centered(0),
    Diffusion(0),
    FileStartCount(0),
    FrameDisplacement(0),
    InContact(0),
    Iterations(1),
    RebindTime(0),
    SaveCounts(0),
    SeparateFiles(0),
    SquaredDisplacement(0),
    Survival(0),
    theFileCnt(0),
    LogEnd(libecs::INF),
    LogStart(0),
    LogInterval(0),
    FileName("IterateLog.csv") {}
  virtual ~IteratingLogProcess() {}
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      SpatiocyteProcess::initialize();
      isPriorityQueued = true;
      theTotalIterations = Iterations;
      theFileCnt = FileStartCount;
    }
  virtual void initializeSecond()
    {
      SpatiocyteProcess::initializeSecond(); 
      for(unsigned int i(0); i != theProcessSpecies.size(); ++i)
        {
          if(InContact)
            {
              theProcessSpecies[i]->setIsInContact();
            }
          if(Centered)
            {
              theProcessSpecies[i]->setIsCentered();
            }
        }
      if(!LogStart)
        {
          LogStart = LogInterval;
        }
      timePointCnt = 0;
      if(!getPriority())
        {
          setPriority(-10);
        }
    }
  virtual void initializeFifth();
  virtual void initializeLastOnce();
  virtual void fire();
  virtual void saveFile();
  virtual void saveBackup();
  virtual void logValues();
  virtual bool isInterrupted(ReactionProcess*) 
    {
      return false;
    }
  void doPreLog();
protected:
  bool isSurviving;
  unsigned Centered;
  unsigned Diffusion;
  unsigned FileStartCount;
  unsigned FrameDisplacement;
  unsigned InContact;
  unsigned Iterations;
  unsigned RebindTime;
  unsigned SaveCounts;
  unsigned SeparateFiles;
  unsigned SquaredDisplacement;
  unsigned Survival;
  unsigned theFileCnt;
  unsigned theTotalIterations;
  unsigned timePoints;
  unsigned timePointCnt;
  double LogEnd;
  double LogStart;
  double LogInterval;
  String FileName;
  std::ofstream theLogFile;
  Comp* theComp;
  std::vector<std::vector<double> > theLogValues;
};

#endif /* __IteratingLogProcess_hpp */
