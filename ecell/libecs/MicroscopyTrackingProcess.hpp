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


#ifndef __MicroscopyTrackingProcess_hpp
#define __MicroscopyTrackingProcess_hpp

#include "VisualizationLogProcess.hpp"
#include "SpatiocyteSpecies.hpp"

LIBECS_DM_CLASS(MicroscopyTrackingProcess, VisualizationLogProcess)
{ 
public:
  LIBECS_DM_OBJECT(MicroscopyTrackingProcess, Process)
    {
      INHERIT_PROPERTIES(VisualizationLogProcess);
      PROPERTYSLOT_SET_GET(Integer, MeanCount);
      PROPERTYSLOT_SET_GET(Real, ExposureTime);
    }
  MicroscopyTrackingProcess():
    MeanCount(0),
    ExposureTime(0.5),
    theLastExposedTime(0)
  {
    FileName = "MicroscopyLog.dat";
  }
  virtual ~MicroscopyTrackingProcess() {}
  SIMPLE_SET_GET_METHOD(Integer, MeanCount);
  SIMPLE_SET_GET_METHOD(Real, ExposureTime);
  virtual void initializeFourth()
    {
      theFreqLatticeSize = theLattice->size();
      //Put all the unique negative species in the lattice species list:
      for(VariableReferenceVector::iterator 
          i(theNegativeVariableReferences.begin());
          i != theNegativeVariableReferences.end(); ++i)
        {
          Variable* aVariable((*i).getVariable());
          Species* aSpecies(theSpatiocyteStepper->getSpecies(aVariable));
          if(!aSpecies->getIsLipid() &&
             std::find(theLatticeSpecies.begin(), theLatticeSpecies.end(), 
                  aSpecies) == theLatticeSpecies.end())
            {
              theLatticeSpecies.push_back(aSpecies);
            }
        }
      thePriority = -1;
    }
  virtual void initializeFifth()
    {
      VariableReferenceVector::iterator aNegativeVariableIter(
                                   theNegativeVariableReferences.begin());
      for(VariableReferenceVector::iterator
          i(thePositiveVariableReferences.begin());
          i != thePositiveVariableReferences.end(); ++i)
        {
          int aPositiveCoefficient((*i).getCoefficient());
          Variable* aVariable((*i).getVariable());
          Species* aSpecies(theSpatiocyteStepper->getSpecies(aVariable));
          if(!aSpecies->getIsLipid())
            {
              if(aPositiveCoefficient > 0)
                {
                  thePositiveSpecies.push_back(aSpecies);
                  std::vector<int> aProcessSpeciesIndices;
                  do
                    {
                      int aNegativeCoefficient(
                               (*aNegativeVariableIter).getCoefficient());
                      Variable* aNegativeVariable(
                                (*aNegativeVariableIter).getVariable());
                      Species* aNegativeSpecies(
                          theSpatiocyteStepper->getSpecies(aNegativeVariable));
                      int aProcessSpeciesIndex(
                       std::find(theLatticeSpecies.begin(), 
                                 theLatticeSpecies.end(), 
                            aNegativeSpecies) - theLatticeSpecies.begin());
                      while(aNegativeCoefficient)
                        {
                          aNegativeCoefficient += 1;
                          aPositiveCoefficient -= 1;
                          aProcessSpeciesIndices.push_back(
                                                   aProcessSpeciesIndex);
                        }
                      ++aNegativeVariableIter;
                    }
                  while(aPositiveCoefficient);
                  theLatticeSpeciesIndices.push_back(aProcessSpeciesIndices);
                }
            }
        }
      theFreqLattice.resize(theLatticeSpecies.size());
      for(unsigned int i(0); i != theFreqLattice.size(); ++i)
        {
          theFreqLattice[i].resize(theFreqLatticeSize);
        }
      resetLattice();
      if(MeanCount > 0)
        {
          theStepInterval = ExposureTime/MeanCount;
        }
      else
        {
          for(std::vector<Species*>::const_iterator
              i(thePositiveSpecies.begin()); i != thePositiveSpecies.end(); ++i)
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
          MeanCount = (int)rint(ExposureTime/theStepInterval);
        }
      theMeanCount = (unsigned int)MeanCount;
      theTime = theStepInterval;
      theLastExposedTime = theTime;
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
      theLogFile.flush();
    }
  virtual void fire()
    {
      incSpeciesLatticeCount();
      if(theTime-theLastExposedTime >= ExposureTime)
        {
          theLastExposedTime = theTime;
          logFluorescentSpecies();
          resetLattice();
        }
      theTime += theStepInterval;
      thePriorityQueue->moveTop();
    }
protected:
  void incSpeciesLatticeCount();
  void logFluorescentSpecies();
  void resetLattice()
    {
      for(unsigned int i(0); i != theFreqLattice.size(); ++i)
        {
          for(unsigned int j(0); j != theFreqLatticeSize; ++j)
            {
              theFreqLattice[i][j] = 0;
            }
        }
    }
protected:
  unsigned int theFreqLatticeSize;
  int MeanCount;
  double ExposureTime;
  double theLastExposedTime;
  std::vector<Species*> thePositiveSpecies;
  std::vector<std::vector<int> > theFreqLattice;
  std::vector<std::vector<int> > theLatticeSpeciesIndices;
};

#endif /* __MicroscopyTrackingProcess_hpp */
