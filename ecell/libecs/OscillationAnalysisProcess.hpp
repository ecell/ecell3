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


#ifndef __OscillationAnalysisProcess_hpp
#define __OscillationAnalysisProcess_hpp

#include <fstream>
#include <SpatiocyteProcess.hpp>
#include <SpatiocyteSpecies.hpp>

LIBECS_DM_CLASS(OscillationAnalysisProcess, SpatiocyteProcess)
{ 
public:
  LIBECS_DM_OBJECT(OscillationAnalysisProcess, Process)
    {
      INHERIT_PROPERTIES(Process);
      PROPERTYSLOT_SET_GET(Real, MembraneBindTime);
      PROPERTYSLOT_SET_GET(Real, LocalizationStartTime);
      PROPERTYSLOT_SET_GET(Real, LocalizationEndTime);
      PROPERTYSLOT_SET_GET(Real, OscillationTime);
      PROPERTYSLOT_SET_GET(Real, AnalysisInterval);
      PROPERTYSLOT_SET_GET(Real, Period);
      PROPERTYSLOT_SET_GET(Integer, Status);
    }
  OscillationAnalysisProcess():
    isLeftNucleated(false),
    isLeftNucleateExited(true),
    isRightNucleated(false),
    isRightNucleateExited(true),
    prevLeftSize(0),
    prevRightSize(0),
    Status(0),
    theCycleCount(0),
    AnalysisInterval(1),
    LocalizationEndTime(13),
    LocalizationEndTime2(5),
    LocalizationStartTime(8),
    LocalizationStartTime2(30),
    MembraneBindTime(6),
    OscillationTime(40) {}
  virtual ~OscillationAnalysisProcess() {}
  SIMPLE_SET_GET_METHOD(Real, MembraneBindTime);
  SIMPLE_SET_GET_METHOD(Real, LocalizationStartTime);
  SIMPLE_SET_GET_METHOD(Real, LocalizationEndTime);
  SIMPLE_SET_GET_METHOD(Real, OscillationTime);
  SIMPLE_SET_GET_METHOD(Real, AnalysisInterval);
  SIMPLE_SET_GET_METHOD(Real, Period);
  SIMPLE_SET_GET_METHOD(Integer, Status);
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      SpatiocyteProcess::initialize();
      isPriorityQueued = true;
    }
  virtual void initializeLastOnce()
    {
      for(unsigned int i(0); i != theProcessSpecies.size(); ++i)
        {
          if(theProcessSpecies[i]->getVariable()->getID() == "MinD")
            {
              minD_m = theProcessSpecies[i];
            }
          else if(theProcessSpecies[i]->getVariable()->getID() == "MinDatp")
            {
              minD = theProcessSpecies[i];
            }
          else if(theProcessSpecies[i]->getVariable()->getID() == "MinEE")
            {
              minE = theProcessSpecies[i];
            }
        }
      theCenterPoint = theSpatiocyteStepper->getCenterPoint();
    }
  virtual GET_METHOD(Real, Interval)
    {
      return MembraneBindTime;
    }
  virtual void fire()
    {
      if(theTime <= MembraneBindTime)
        {
          testMembraneBinding();
          theTime = LocalizationStartTime;
        }
      else if(theTime >= LocalizationStartTime &&
              theTime <= LocalizationEndTime )
        {
          testLocalization(2);
          theTime += AnalysisInterval;
          if(theTime > LocalizationEndTime)
            {
              theTime = LocalizationStartTime2;
            }
        }
      else if(theTime >= LocalizationStartTime2 &&
              theTime <= LocalizationEndTime2 )
        {
          testLocalization(3);
          theTime += AnalysisInterval;
          if(theTime > LocalizationEndTime2)
            {
              theTime = OscillationTime;
            }
        }
      else
        {
          testOscillation();
          theTime += AnalysisInterval;
        }
      thePriorityQueue->moveTop();
    }
protected:
  virtual void testMembraneBinding();
  virtual void testLocalization(int);
  virtual void testOscillation();
protected:
  bool isLeftNucleated;
  bool isLeftNucleateExited;
  bool isRightNucleated;
  bool isRightNucleateExited;
  unsigned int prevLeftSize;
  unsigned int prevRightSize;
  int currStatus;
  int prevLeftStatus;
  int prevRightStatus;
  int Status;
  int theCycleCount;
  double AnalysisInterval;
  double LocalizationEndTime;
  double LocalizationEndTime2;
  double LocalizationStartTime;
  double LocalizationStartTime2;
  double MembraneBindTime;
  double OscillationTime;
  double Period;
  double theLeftBeginTime; 
  double theRightBeginTime; 
  double theTotalPeriod;
  Species* minD;
  Species* minD_m;
  Species* minE;
  Point theCenterPoint;
};

#endif /* __OscillationAnalysisProcess_hpp */
