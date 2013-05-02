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

#include <algorithm>
#include <gsl/gsl_randist.h>
#include <boost/lexical_cast.hpp>
#include <libecs/Model.hpp>
#include <libecs/System.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/Process.hpp>
#include <libecs/VariableReference.hpp>
#include <TagProcess.hpp>
#include <SpatiocyteSpecies.hpp>
#include <SpatiocyteProcess.hpp>

LIBECS_DM_INIT(TagProcess, Process); 


void TagProcess::initialize()
{
  if(isInitialized)
    {
      return;
    }
  SpatiocyteProcess::initialize();
  for(VariableReferenceVector::const_iterator
      i(theVariableReferenceVector.begin());
      i != theVariableReferenceVector.end(); ++i)
    {
      Variable* aVariable((*i).getVariable());
      if(aVariable->getName() == "HD")
        {
          THROW_EXCEPTION(ValueError, String(
                      getPropertyInterface().getClassName()) +
                      "[" + getFullID().asString() + "]: All variable " +
                      "references of a TagProcess must be nonHD species, but " +
                      getIDString(aVariable) + ", which is a HD species is " +
                      "given.");
        }
    }
  if(theNegativeVariableReferences.size() > 1)
    { 
      THROW_EXCEPTION(ValueError, String(
                      getPropertyInterface().getClassName()) +
                      "[" + getFullID().asString() + "]: A TagProcess " + 
                      "requires only one negative variable reference that " +
                      "will specify the tag, but more than one is given.");
    }
  if(theNegativeVariableReferences.size() == 0)
    { 
      THROW_EXCEPTION(ValueError, String(
                      getPropertyInterface().getClassName()) +
                      "[" + getFullID().asString() + "]: A TagProcess " + 
                      "requires one negative variable reference that " +
                      "will specify the tag, but none is given.");
    }
  theTagSpecies = theSpatiocyteStepper->variable2species(
                       theNegativeVariableReferences[0].getVariable());
  for(VariableReferenceVector::const_iterator
      i(theVariableReferenceVector.begin());
      i != theVariableReferenceVector.end(); ++i)
    {
      int aCoefficient((*i).getCoefficient());
      //All positive and zero VariableReferences are tagged species:
      if(aCoefficient >= 0)
        {
          theTaggedSizes.push_back(aCoefficient);
          Variable* aVariable((*i).getVariable());
          Species* aSpecies(theSpatiocyteStepper->variable2species(aVariable)); 
          if(aSpecies == theTagSpecies)
            {
              THROW_EXCEPTION(ValueError, String(
                              getPropertyInterface().getClassName()) +
                              "[" + getFullID().asString() + "]: The species " +
                              getIDString(aSpecies) + " is the tag species. " +
                              "We cannot specify the same species as a " +
                              "tagged species.");
            }
          if(std::find(theTaggedSpeciesList.begin(), theTaggedSpeciesList.end(),
                       aSpecies) != theTaggedSpeciesList.end())
            {
              THROW_EXCEPTION(ValueError, String(
                              getPropertyInterface().getClassName()) +
                              "[" + getFullID().asString() + "]: Duplicate " +
                              "declaration of species " + 
                              getIDString(aSpecies) + " as a tagged species.");
            }
          theTaggedSpeciesList.push_back(aSpecies);
        }
    }
}

void TagProcess::initializeSecond()
{
  SpatiocyteProcess::initializeSecond();
  //We can only do this is initializeSecond because off lattice species
  //are only defined in initializeFirst by other processes:
  for(unsigned i(0); i != theTaggedSpeciesList.size(); ++i)
    {
      Species* aSpecies(theTaggedSpeciesList[i]);
      aSpecies->addTagSpecies(theTagSpecies);
    }
}

void TagProcess::initializeFourth()
{
  for(unsigned i(0); i != theTaggedSpeciesList.size(); ++i)
    { 
      Species* aSpecies(theTaggedSpeciesList[i]);
      unsigned anAvailableSize(0);
      for(unsigned j(0); j != aSpecies->size(); ++j)
        {
          if(aSpecies->getTagID(j) == theNullID)
            {
              ++anAvailableSize;
              if(anAvailableSize >= theTaggedSizes[i])
                {
                  break;
                }
            }
        }
      if(anAvailableSize < theTaggedSizes[i])
        {
          THROW_EXCEPTION(ValueError, String(
                          getPropertyInterface().getClassName()) +
                          "[" + getFullID().asString() + "]: The number of " +
                          "available molecules of the tagged species " +
                          getIDString(aSpecies) + " is " + 
                          int2str(anAvailableSize) + ", but there are " + 
                          int2str(theTaggedSizes[i]) + " molecules that must " +
                          "be tagged by the tag species " +
                          getIDString(theTagSpecies) + ". Please increase " +
                          "the molecule number of tagged species or reduce " +
                          "its tagged coefficient value in this process.");
        }
      for(unsigned j(0); j != theTaggedSizes[i]; ++j)
        {
          unsigned anIndex(aSpecies->getRandomIndex());
          while(aSpecies->getTagID(anIndex) != theNullID)
            {
              anIndex = aSpecies->getRandomIndex();
            }
          aSpecies->setTagID(anIndex, theTagSpecies->getID());
        }
    }
  theTagSpecies->updateMolecules();
  theTagSpecies->setIsPopulated();
}


