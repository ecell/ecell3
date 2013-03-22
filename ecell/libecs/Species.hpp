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


#ifndef __Species_hpp
#define __Species_hpp

#include <sstream>
#include <algorithm>
#include <libecs/Variable.hpp>
#include <gsl/gsl_randist.h>
#include "RandomLib/Random.hpp"
#include "RandomLib/Random.cpp"
#include "SpatiocyteCommon.hpp"
#include "SpatiocyteStepper.hpp"
#include "SpatiocyteProcessInterface.hpp"
#include "SpatiocyteNextReactionProcess.hpp"
#include "DiffusionInfluencedReactionProcess.hpp"
#include "MoleculePopulateProcessInterface.hpp"
#include "Thread.hpp"

static double getDistance(Point& aSourcePoint, Point& aDestPoint)
{
  return sqrt(pow(aDestPoint.x-aSourcePoint.x, 2)+
              pow(aDestPoint.y-aSourcePoint.y, 2)+
              pow(aDestPoint.z-aSourcePoint.z, 2));
}

String int2str(int anInt)
{
  std::stringstream aStream;
  aStream << anInt;
  return aStream.str();
}

/*
 * isVacant: vacant species definition: a species on which other species can
 * diffuse on or occupy. One major optimization in speed for vacant species is
 * that theMols list is not updated when other species diffuse on it. 
 * There are four possible types of vacant species:
 * 1. !isDiffusiveVacant && !isReactiveVacant: a vacant species
 *    whose theMols list are never updated. This is
 *    the most basic version. This vacant species never diffuses, or reacts
 *    using SNRP. 
 * 2. !isDiffusiveVacant && isReactiveVacant: a vacant species which is a
 *    substrate of a SNRP reaction. In this case theMolSize is updated
 *    before the step interval of SNRP is calculated, and theMols list is
 *    updated before the SNRP reaction (fire) is executed to get a valid
 *    molecule list. Set by SNRP.
 * 3. isDiffusiveVacant && !isReactiveVacant: a vacant species which also
 *    diffuses. In this case, theMols list and theMolSize are
 *    updated just before it is diffused. Set by DiffusionProcess.   
 * 4. isDiffusiveVacant && isReactiveVacant: a vacant species which reacts
 *    using SNRP and also diffuses.
 * 5. isCompVacant: the VACANT species declared for each compartment. It can
 *    also be isDiffusiveVacant and isReactiveVacant. Set during compartment
 *    registration. It also persistently stores all the compartment voxels.
 *    Referred to as theVacantSpecies. For isCompVacant, initially there
 *    are no molecules in its list. All voxels are stored in theCompVoxels. Only
 *    if it is updated before being called by VisualizationLogProcess, SNRP or
 *    DiffusionProcess, it will be theMols will be populated with the
 *    CompMols.
 * 6. isVacant {isCompVacant; isDiffusiveVacant; isReactiveVacant): the
 *    general name used to identify either isCompVacant, isDiffusiveVacant or
 *    isReactiveVacant to reduce comparison operations.
 */

class Species
{
public:
  Species(SpatiocyteStepper* aStepper, Variable* aVariable,
          unsigned short anID, const gsl_rng* aRng,
          double voxelRadius, std::vector<std::vector<unsigned short> >& anIDs,
          std::vector<std::vector<VoxelInfo> >& anInfo,
          std::vector<std::vector<unsigned> >& anAdjoins,
          std::vector<std::vector<unsigned> >& anAdjBoxes,
          std::vector<std::vector<unsigned> >& anAdjAdjBoxes):
    isCentered(false),
    isCompVacant(false),
    isDiffusing(false),
    isDiffusiveVacant(false),
    isFixedAdjoins(false),
    isGaussianPopulation(false),
    isInContact(false),
    isMultiscale(false),
    isOffLattice(false),
    isPolymer(false),
    isReactiveVacant(false),
    isSubunitInitialized(false),
    isTag(false),
    isTagged(false),
    isVacant(false),
    theID(anID),
    theCollision(0),
    D(0),
    theDiffuseRadius(voxelRadius),
    theDiffusionInterval(libecs::INF),
    theMolRadius(voxelRadius),
    theVoxelRadius(voxelRadius),
    theWalkProbability(1),
    theRng(gsl_rng_alloc(gsl_rng_ranlxs2)),
    thePopulateProcess(NULL),
    theStepper(aStepper),
    theVariable(aVariable),
    theCompMols(&theMols),
    theAdjoins(anAdjoins),
    theAdjBoxes(anAdjBoxes),
    theAdjAdjBoxes(anAdjAdjBoxes),
    theInfo(anInfo),
    theIDs(anIDs) {}
  ~Species() {}
  void initialize(int speciesSize, unsigned aBoxMaxSize, int anAdjoinSize,
                  unsigned aNullMol, unsigned aNullID, 
                  std::vector<Thread*> aThreads)
    {
      theThreads = aThreads;
      theThreadSize = theThreads.size();
      theBoxMaxSize = aBoxMaxSize;
      theBoxSize = theIDs.size();
      theLastMolSize.resize(theBoxSize);
      theCnt.resize(theBoxSize);
      theMols.resize(theBoxSize);
      theTars.resize(theBoxSize);
      theRands.resize(theBoxSize);
      theInitMolSize.resize(theBoxSize);
      theTags.resize(theBoxSize);
      theRepeatAdjMols.resize(theBoxSize);
      theRepeatAdjTars.resize(theBoxSize);
      rng.Reseed();
      unsigned anInitMolSize(0);
      if(getVariable())
        {
          anInitMolSize = getVariable()->getValue();
        }
      unsigned bal(anInitMolSize%theBoxSize);
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          theInitMolSize[i] = anInitMolSize/theBoxSize;
          theRepeatAdjMols[i].resize(theBoxSize);
          theRepeatAdjTars[i].resize(theBoxSize);
          if(bal)
            {
              theInitMolSize[i] += 1;
              --bal;
            }
        }
      theAdjMols.resize(2);
      theAdjTars.resize(2);
      theAdjAdjMols.resize(2);
      theAdjAdjTars.resize(2);
      theBorderMols.resize(2);
      theBorderTars.resize(2);
      for(unsigned i(0); i != 2; ++i)
        {
          theAdjMols[i].resize(theBoxSize);
          theAdjTars[i].resize(theBoxSize);
          theAdjAdjMols[i].resize(theBoxSize);
          theAdjAdjTars[i].resize(theBoxSize);
          theBorderMols[i].resize(theBoxSize);
          theBorderTars[i].resize(theBoxSize);
          for(unsigned j(0); j != theBoxSize; ++j)
            {
              theAdjMols[i][j].resize(theBoxSize);
              theAdjTars[i][j].resize(theBoxSize);
              theAdjAdjMols[i][j].resize(theBoxSize);
              theAdjAdjTars[i][j].resize(theBoxSize);
              theBorderMols[i][j].resize(theBoxSize);
              theBorderTars[i][j].resize(theBoxSize);
            }
        }
      theAdjoinSize = anAdjoinSize;
      theNullMol = aNullMol;
      theNullID = aNullID;
      theSpeciesSize = speciesSize;
      theReactionProbabilities.resize(speciesSize);
      theDiffusionInfluencedReactions.resize(speciesSize);
      theFinalizeReactions.resize(speciesSize);
      theMultiscaleBindIDs.resize(speciesSize);
      theMultiscaleUnbindIDs.resize(speciesSize);
      for(int i(0); i != speciesSize; ++i)
        {
          theDiffusionInfluencedReactions[i] = NULL;
          theReactionProbabilities[i] = 0;
          theFinalizeReactions[i] = false;
        }
      if(theComp)
        {
          setVacantSpecies(theComp->vacantSpecies);
        }
      theNullTag.origin = theNullMol;
      theNullTag.id = theNullID;
    }
  void setDiffusionInfluencedReaction(
                                    DiffusionInfluencedReactionProcess*
                                      aReaction, int anID, double aProbability)
    {
      theDiffusionInfluencedReactions[anID] = aReaction;
      theReactionProbabilities[anID] = aProbability;
    }
  void setDiffusionInfluencedReactantPair(Species* aSpecies)
    {
      theDiffusionInfluencedReactantPairs.push_back(aSpecies);
    }
  void setPopulateProcess(MoleculePopulateProcessInterface* aProcess,
                          double aDist)
    {
      if(aDist)
        {
          isGaussianPopulation = true;
        }
      thePopulateProcess = aProcess;
    }
  bool getIsGaussianPopulation()
    {
      return isGaussianPopulation;
    }
  int getPopulatePriority()
    {
      return thePopulateProcess->getPriority();
    }
  void populateCompGaussian()
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateGaussian(this);
        }
      else if(size())
        {
          std::cout << "Warning: Species " << getIDString() <<
            " not populated." << std::endl;
        }
    }
  void addTaggedSpecies(Species* aSpecies)
    {
      isTag = true;
      //If one of the tagged species is off-lattice then
      //make the tag species off-lattice. The getPoint method
      //will convert the coord of lattice species to point
      //when logging:
      if(aSpecies->getIsOffLattice())
        {
          isOffLattice = true;
        }
      theTaggedSpeciesList.push_back(aSpecies);
    }
  void addTagSpecies(Species* aSpecies)
    {
      isTagged = true;
      theTagSpeciesList.push_back(aSpecies);
      aSpecies->addTaggedSpecies(this);
    }
  bool getIsTagged()
    {
      return isTagged;
    }
  bool getIsTag()
    {
      return isTag;
    }
  bool getIsPopulateSpecies()
    {
      return (thePopulateProcess != NULL);
    }
  void populateCompUniform(unsigned* voxelIDs, unsigned* aCount)
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformDense(this, voxelIDs, aCount);
        }
      else if(size())
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not MoleculePopulated." << std::endl;
        }
    }
  void populateCompUniformSparse()
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformSparse(this);
        }
      else if(size())
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not MoleculePopulated." << std::endl;
        }
    }
  void populateUniformOnDiffusiveVacant()
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformOnDiffusiveVacant(this);
        }
      else if(size())
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not MoleculePopulated." << std::endl;
        }
    }
  void populateUniformOnMultiscale()
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformOnMultiscale(this);
        }
      else if(size())
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not MoleculePopulated." << std::endl;
        }
    }
  Variable* getVariable() const
    {
      return theVariable;
    }
  unsigned size(unsigned aBox) const
    {
      return theMols[aBox].size();
    }
  unsigned size() const
    {
      unsigned aSize(0);
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          aSize += size(i);
        }
      return aSize;
    }
  unsigned& getMol(unsigned aBox, unsigned anIndex)
    {
      return theMols[aBox][anIndex];
    }
  Point& getPoint(unsigned aBox, unsigned anIndex)
    {
      return theInfo[aBox][getMol(aBox, anIndex)-aBox*theBoxMaxSize].point;
    }
  unsigned short getID() const
    {
      return theID;
    }
  double getMeanSquaredDisplacement()
    {
      const unsigned aSize(size());
      if(aSize)
        {
          return 0;
        }
      double aDisplacement(0);
      /*
      for(unsigned i(0); i < aSize; ++i)
        {
          Point aCurrentPoint(theStepper->getPeriodicPoint(
                                                 getMol(i),
                                                 theDimension,
                                                 &theMolOrigins[i]));
          double aDistance(getDistance(theMolOrigins[i].point,
                                       aCurrentPoint));
          aDisplacement += aDistance*aDistance;
        }
        */
      return aDisplacement*pow(theDiffuseRadius*2, 2)/aSize;
    }
  void setCollision(unsigned aCollision)
    {
      theCollision = aCollision;
    }
  void setIsSubunitInitialized()
    {
      isSubunitInitialized = true;
    }
  void setIsMultiscale()
    {
      isMultiscale = true;
    }
  bool getIsMultiscale()
    {
      return isMultiscale;
    }
  void setIsCompVacant()
    {
      isCompVacant = true;
      isVacant = true;
      setVacantSpecies(this);
    }
  void setIsInContact()
    {
      isInContact = true;
    }
  void setIsCentered()
    {
      isCentered = true;
    }
  void setIsPopulated()
    {
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          theInitMolSize[i] = theMols[i].size();
        }
      getVariable()->setValue(size());
    }
  void finalizeSpecies()
    {
      if(theCollision)
        {
          collisionCnts.resize(size());
          for(std::vector<unsigned>::iterator 
              i(collisionCnts.begin()); i != collisionCnts.end(); ++i)
            {
              *i = 0;
            }
        }
      //need to shuffle molecules of the compVacant species if it has
      //diffusing vacant species to avoid bias when random walking:
      if(isCompVacant)
        {
          /*
          for(unsigned i(0); i != theComp->species.size(); ++i)
            {
              if(theComp->species[i]->getIsDiffusiveVacant())
                {
                  std::random_shuffle(theMols.begin(), theMols.end());
                  break;
                }
            }
            */
          if(isTagged)
            {
              for(unsigned i(0); i != theBoxSize; ++i)
                {
                  for(unsigned j(0); j != theMols[i].size(); ++j)
                    {
                      theTags[i][j].origin = getMol(i, j);
                    }
                }
            }
        }
      isToggled = false;
    }
  void initializeLists(unsigned anID, 
                       RandomLib::Random& aRng,
                       std::vector<unsigned>& aMols,
                       std::vector<unsigned>& aTars,
                       std::vector<std::vector<std::vector<unsigned> > >& anAdjMols,
                       std::vector<std::vector<std::vector<unsigned> > >& anAdjTars,
                       std::vector<unsigned>& anAdjoins,
                       std::vector<unsigned short>& anIDs,
                       std::vector<unsigned>& anAdjBoxes,
                       std::vector<unsigned>& anAdjAdjBoxes,
                       std::vector<unsigned>& aRands)
    {
      if(theDiffusionInterval != libecs::INF)
        {
          std::cout << "Initializing:" << anID << std::endl;
          aMols.resize(theMols[anID].size());
          for(unsigned i(0); i != theMols[anID].size(); ++i)
            {
              aMols[i] = theMols[anID][i];
            }
          anAdjoins.resize(theAdjoins[anID].size());
          for(unsigned i(0); i != theAdjoins[anID].size(); ++i)
            {
              anAdjoins[i] = theAdjoins[anID][i];
            } 
          anAdjBoxes.resize(theAdjBoxes[anID].size());
          for(unsigned i(0); i != theAdjBoxes[anID].size(); ++i)
            {
              anAdjBoxes[i] = theAdjBoxes[anID][i];
            } 
          anAdjAdjBoxes.resize(theAdjAdjBoxes[anID].size());
          for(unsigned i(0); i != theAdjAdjBoxes[anID].size(); ++i)
            {
              anAdjAdjBoxes[i] = theAdjAdjBoxes[anID][i];
            } 
          anIDs.resize(theIDs[anID].size());
          for(unsigned i(0); i != theIDs[anID].size(); ++i)
            {
              anIDs[i] = theIDs[anID][i];
            } 
          setRands(anID, 1, aMols.size(), anAdjBoxes, aRands, aRng);
          setTars(anID, aMols, aTars, anAdjMols[0], anAdjTars[0], anAdjoins, aRands);
          //setTars(theMols[anID], theTars[anID], theAdjMols[0], theAdjTars[0], anID, theAdjoins[anID]);
        }
    }
  unsigned getCollisionCnt(unsigned anIndex)
    {
      return collisionCnts[anIndex];
    }
  unsigned getCollision() const
    {
      return theCollision;
    }
  void setIsDiffusiveVacant()
    {
      isDiffusiveVacant = true;
      isVacant = true;
    }
  void setIsReactiveVacant()
    {
      isReactiveVacant = true;
      isVacant = true;
    }
  void setIsOffLattice()
    {
      isOffLattice = true;
    }
  void resetFixedAdjoins()
    {
      isFixedAdjoins = false;
      for(unsigned i(0); i != theComp->species.size(); ++i)
        {
          theComp->species[i]->setIsFixedAdjoins(false);
        }
    }
  void setIsFixedAdjoins(bool state)
    {
      isFixedAdjoins = state;
    }
  void setIsPolymer(std::vector<double> bendAngles, int aDirectionality)
    {
      theBendAngles.resize(0);
      thePolymerDirectionality = aDirectionality;
      isPolymer = true;
      for(std::vector<double>::const_iterator i(bendAngles.begin()); 
          i != bendAngles.end(); ++i)
        {
          theBendAngles.push_back(*i);
          if(thePolymerDirectionality != 0)
            {
              theBendAngles.push_back(*i+M_PI);
            }
        }
    }
  void setDiffusionCoefficient(double aCoefficient)
    {
      D = aCoefficient;
      if(D > 0)
        {
          isDiffusing = true;
        }
    }
  double getDiffusionCoefficient() const
    {
      return D;
    }
  double getWalkProbability() const
    {
      return theWalkProbability;
    }
  bool getIsPolymer() const
    {
      return isPolymer;
    }
  bool getIsOffLattice()
    {
      return isOffLattice;
    }
  bool getIsSubunitInitialized() const
    {
      return isSubunitInitialized;
    }
  bool getIsDiffusing() const
    {
      return isDiffusing;
    }
  bool getIsCompVacant() const
    {
      return isCompVacant;
    }
  bool getIsVacant() const
    {
      return isVacant;
    }
  bool getIsDiffusiveVacant()
    {
      return isDiffusiveVacant;
    }
  bool getIsReactiveVacant()
    {
      return isReactiveVacant;
    }
  bool getIsLipid() const
    {
      return (isCompVacant && theComp->dimension == 2);
    }
  bool getIsInContact() const
    {
      return isInContact;
    }
  bool getIsCentered() const
    {
      return isCentered;
    }
  bool getIsPopulated() const
    {
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          if(theInitMolSize[i] != theMols[i].size())
            {
              return false;
            }
        }
      return true;
    }
  double getDiffusionInterval() const
    {
      return theDiffusionInterval;
    }
  unsigned getDimension()
    {
      return theDimension;
    }
  void setDimension(unsigned aDimension)
    {
      theDimension = aDimension;
      if(theDimension == 3)
        {
          isFixedAdjoins = true;
        }
    }
  void resetFinalizeReactions()
    {
      for(unsigned i(0); i != theFinalizeReactions.size(); ++i)
        {
          theFinalizeReactions[i] = false;
        }
    }
  void finalizeReactions()
    {
      for(unsigned i(0); i != theFinalizeReactions.size(); ++i)
        {
          if(theFinalizeReactions[i])
            {
              theDiffusionInfluencedReactions[i]->finalizeReaction();
            }
        }
      for(unsigned i(0); i != theInterruptedProcesses.size(); ++i)
        {
          theInterruptedProcesses[i
            ]->substrateValueChanged(theStepper->getCurrentTime());
        }
    }
  void addCollision(unsigned aMol)
    {
      /*
      for(unsigned i(0); i < theMolSize; ++i)
        {
          if(theMols[i] == aMol)
            {
              ++collisionCnts[i];
              return;
            }
        }
      std::cout << "error in species add collision" << std::endl;
      */
    }
  void setRands(const unsigned currBox,
                       const unsigned r,
                       unsigned aSize,
                       const std::vector<unsigned>& anAdjBoxes,
                       std::vector<unsigned>& aRands,
                       RandomLib::Random& aRng);
  void walkMols(std::vector<unsigned>& aMols,
                const std::vector<unsigned>& aTars,
                std::vector<unsigned short>& anIDs);
  void walkAdjMols(const unsigned currBox, const unsigned r,
                          std::vector<unsigned>& aMols,
                          std::vector<unsigned short>& anIDs,
                          const std::vector<unsigned>& anAdjBoxes);
  void setAdjTars(const unsigned currBox, const unsigned r,
                std::vector<std::vector<unsigned> >& aBorderMols,
                std::vector<std::vector<unsigned> >& aBorderTars,
                std::vector<std::vector<unsigned> >& anAdjAdjMols,
                std::vector<std::vector<unsigned> >& anAdjAdjTars,
                std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                const std::vector<unsigned>& anAdjBoxes,
                unsigned startRand, std::vector<unsigned>& aRands);
  void updateAdjMols(const unsigned currBox, const unsigned r,
                     std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                     std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                     const std::vector<unsigned>& anAdjBoxes);
  void updateAdjAdjMols(const unsigned currBox, const unsigned r,
                        const std::vector<unsigned>& anAdjAdjBoxes);
  void setTars(const unsigned currBox,
               std::vector<unsigned>& aMols,
               std::vector<unsigned>& aTars,
               std::vector<std::vector<unsigned> >&,
               std::vector<std::vector<unsigned> >&,
               const std::vector<unsigned>& anAdjoins,
               std::vector<unsigned>& aRands);
  void updateBoxMols(const unsigned currBox, const unsigned r,
                   std::vector<unsigned>& aMols,
                   std::vector<unsigned>& aTars,
                   const std::vector<unsigned>& anAdjBoxes);
  void walk(const unsigned anID, unsigned r, unsigned w,
           RandomLib::Random& aRng,
           std::vector<unsigned>& aMols,
           std::vector<unsigned>& aTars,
           std::vector<std::vector<std::vector<unsigned> > >& anAdjMols,
           std::vector<std::vector<std::vector<unsigned> > >& anAdjTars,
           std::vector<std::vector<std::vector<unsigned> > >& anAdjAdjMols,
           std::vector<std::vector<std::vector<unsigned> > >& anAdjAdjTars,
           std::vector<std::vector<std::vector<unsigned> > >& aBorderMols,
           std::vector<std::vector<std::vector<unsigned> > >& aBorderTars,
           std::vector<std::vector<unsigned> >& aRepeatAdjMols,
           std::vector<std::vector<unsigned> >& aRepeatAdjTars,
           const std::vector<unsigned>& anAdjoins,
           std::vector<unsigned short>& anIDs,
           const std::vector<unsigned>& anAdjBoxes,
           const std::vector<unsigned>& anAdjAdjBoxes,
           std::vector<unsigned>& aRands);
  /*
  void setTars(std::vector<unsigned>& aMols,
               std::vector<unsigned>& aTars,
               std::vector<std::vector<std::vector<unsigned> > >& aNextMols,
               std::vector<std::vector<std::vector<unsigned> > >& aNextTars,
               const unsigned currBox,
               const std::vector<unsigned>& anAdjoins);
  void walkMols(std::vector<unsigned>& aMols,
                const std::vector<unsigned>& aTars,
                std::vector<unsigned short>& anIDs);
  void walkAdjMols(std::vector<unsigned>& aMols,
                   std::vector<std::vector<unsigned> >& aBoxAdjMols,
                   std::vector<std::vector<unsigned> >& aBoxAdjTars, 
                   std::vector<unsigned short>& anIDs,
                   const std::vector<unsigned>& anAdjBoxes);
  void setAdjTars(std::vector<std::vector<std::vector<unsigned> > >&
                  aBorderMols,
                  std::vector<std::vector<std::vector<unsigned> > >&
                  aBorderTars,
                  std::vector<std::vector<std::vector<unsigned> > >&
                  anAdjAdjMols,
                  std::vector<std::vector<std::vector<unsigned> > >&
                  anAdjAdjTars,
                  std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                  std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                  std::vector<std::vector<unsigned> >& anAdjMols,
                  const std::vector<unsigned>& anAdjBoxes,
                  const unsigned currBox);
  void updateBoxMols(std::vector<std::vector<unsigned> >& aBorderMols,
                     std::vector<std::vector<unsigned> >& aBorderTars,
                     std::vector<unsigned>& aMols,
                     std::vector<unsigned>& aTars,
                     const std::vector<unsigned>& anAdjBoxes);
  void updateAdjMols(std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                     std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                     std::vector<std::vector<unsigned> >& anAdjMols,
                     std::vector<std::vector<unsigned> >& anAdjTars,
                     const std::vector<unsigned>& anAdjBoxes);
  void updateAdjAdjMols(std::vector<std::vector<unsigned> >& anAdjAdjMols,
                        std::vector<std::vector<unsigned> >& anAdjAdjTars,
                        std::vector<std::vector<unsigned> >& anAdjMols,
                        std::vector<std::vector<unsigned> >& anAdjTars);
                        */

  void walkMultiscale()
    {
      /*
      unsigned beginMolSize(theMolSize);
      for(unsigned i(0); i < beginMolSize && i < theMolSize; ++i)
        {
          unsigned srcMol(theMols[i]);
          unsigned short& source(theIDs[srcMol]);
          int size(theInfo[srcMol].diffuseSize);
          unsigned tarMol(theAdjoins[theMols[i]*theAdjoinSize+gsl_rng_uniform_int(theRng, size)]);
          unsigned short& target(theIDs[tarMol]);
          if(target == theVacantID)
            {
              if(theWalkProbability == 1 ||
                 gsl_rng_uniform(theRng) < theWalkProbability)
                {
                  if(!isIntersectMultiscale(srcMol, tarMol))
                    {
                      removeMultiscaleMol(srcMol);
                      addMultiscaleMol(tarMol);
                      target = theID;
                      source = theVacantID;
                      theMols[i] = tarMol;
                    }
                }
            }
        }
        */
    }
  void walkVacant()
    {
      /*
      updateVacantMols();
      for(unsigned i(0); i < theMolSize; ++i)
        {
          unsigned short& source(theIDs[theMols[i]]);
          int size;
          if(isFixedAdjoins)
            {
              size = theAdjoinSize;
            }
          else
            {
              size = theInfo[theMols[i]].diffuseSize;
            }
          unsigned aMol(theAdjoins[theMols[i]*theAdjoinSize+
                        gsl_rng_uniform_int(theRng, size)]);
          unsigned short& target(theIDs[aMol]);
          if(target == theVacantID)
            {
              if(theWalkProbability == 1 ||
                 gsl_rng_uniform(theRng) < theWalkProbability)
                {
                  target = theID;
                  source = theVacantID;
                  theMols[i] = aMol;
                }
            }
        }
        */
    }
  void react(unsigned srcMol, unsigned tarMol, unsigned sourceIndex,
             unsigned targetIndex, Species* targetSpecies)
    {
      /*
      DiffusionInfluencedReactionProcess* aReaction(
               theDiffusionInfluencedReactions[targetSpecies->getID()]);
      unsigned moleculeA(srcMol);
      unsigned moleculeB(tarMol);
      unsigned indexA(sourceIndex);
      unsigned indexB(targetIndex);
      if(aReaction->getA() != this)
        {
          indexA = targetIndex; 
          indexB = sourceIndex;
          moleculeA = tarMol;
          moleculeB = srcMol;
        }
      if(aReaction->react(moleculeA, moleculeB, indexA, indexB))
        {
          //Soft remove the source molecule, i.e., keep the id:
          softRemoveMolIndex(sourceIndex);
          //Soft remove the target molecule:
          //Make sure the targetIndex is valid:
          //Target and Source are same species:
          //For some reason if I use theMols[sourceIndex] instead
          //of getMol(sourceIndex) the walk method becomes
          //much slower when it is only diffusing without reacting:
          if(targetSpecies == this && getMol(sourceIndex) == tarMol)
            {
              softRemoveMolIndex(sourceIndex);
            }
          //If the targetSpecies is a multiscale species with implicit
          //molecule, theTargetIndex is equal to the target molecule size,
          //so we use this info to avoid removing the implicit target molecule:
          else if(targetIndex != targetSpecies->size())
            {
              targetSpecies->softRemoveMolIndex(targetIndex);
            }
          theFinalizeReactions[targetSpecies->getID()] = true;
        }
        */
    }
  void setComp(Comp* aComp)
    {
      theComp = aComp;
    }
  Comp* getComp()
    {
      return theComp;
    }
  void setVariable(Variable* aVariable)
    {
      theVariable = aVariable;
    }
  void removeSurfaces()
    {
      /*
      int newMolSize(0);
      for(unsigned i(0); i < theMolSize; ++i) 
        {
          unsigned aMol(getMol(i));
          if(theStepper->isRemovableEdgeMol(aMol, theComp))
            {
              Comp* aSuperComp(
                 theStepper->system2Comp(theComp->system->getSuperSystem())); 
              aSuperComp->vacantSpecies->addCompVoxel(aMol);
            }
          else 
            { 
              theMols[newMolSize] = aMol;
              ++newMolSize; 
            }
        }
      theMolSize = newMolSize;
      //Must resize, otherwise compVoxelSize will be inaccurate:
      theMols.resize(theMolSize);
      theVariable->setValue(theMolSize);
      */
    }
  void removePeriodicEdgeVoxels()
    {
      /*
      int newMolSize(0);
      for(unsigned i(0); i < theMolSize; ++i) 
        {
          unsigned aMol(getMol(i));
          if(theStepper->isPeriodicEdgeMol(aMol, theComp))
            {
              theIDs[aMol] = theIDs[theNullMol];
            }
          else 
            { 
              theMols[newMolSize] = aMol;
              ++newMolSize; 
            }
        }
      theMolSize = newMolSize;
      //Must resize, otherwise compVoxelSize will be inaccurate:
      theMols.resize(theMolSize);
      theVariable->setValue(theMolSize);
      */
    }
  void updateSpecies()
    {
      /*
      if(isCompVacant && (isDiffusiveVacant || isReactiveVacant))
        {
          theCompMols = new std::vector<unsigned>;
          for(unsigned i(0); i != theMolSize; ++i)
            { 
              theCompMols->push_back(theMols[i]);
            }
        }
        */
    }
  //If it isReactiveVacant it will only be called by SNRP when it is substrate
  //If it isDiffusiveVacant it will only be called by DiffusionProcess before
  //being diffused. So we need to only check if it isVacant:
  void updateMols();
  //If it isReactiveVacant it will only be called by SNRP when it is substrate:
  void updateMolSize()
    {
      if(isDiffusiveVacant || isReactiveVacant)
        {
          updateVacantMolSize();
        }
    }
  void updateTagMols()
    {
      /*
      theMolSize = 0;
      for(unsigned i(0); i != theTaggedSpeciesList.size(); ++i)
        {
          Species* aSpecies(theTaggedSpeciesList[i]);
          for(unsigned j(0); j != aSpecies->size(); ++j)
            {
              if(aSpecies->getTagID(j) == theID)
                {
                  unsigned aMol(aSpecies->getMol(j));
                  ++theMolSize;
                  if(theMolSize > theMols.size())
                    {
                      theMols.push_back(aMol);
                    }
                  else
                    {
                      theMols[theMolSize-1] = aMol;
                    }
                }
            }
        }
        */
    }
  //Even if it is a isCompVacant, this method will be called by
  //VisualizationLogProcess, or SNRP if it is Reactive, or DiffusionProcess
  //if it is Diffusive:
  void updateVacantMols()
    {
      /*
      theMolSize = 0;
      int aSize(theVacantSpecies->compVoxelSize());
      for(int i(0); i != aSize; ++i)
        { 
          //Voxel* aVoxel(theVacantSpecies->getCompVoxel(i));
          unsigned aMol(theVacantSpecies->getCompMol(i));
          //if(aVoxel->id == theID)
          if(theIDs[aMol] == theID)
            {
              ++theMolSize;
              if(theMolSize > theMols.size())
                {
                  theMols.push_back(aMol);
                }
              else
                {
                  theMols[theMolSize-1] = aMol;
                }
            }
        }
      theVariable->setValue(theMolSize);
      */
    }
  void updateVacantMolSize()
    {
      /*
      theMolSize = 0;
      int aSize(theVacantSpecies->compVoxelSize());
      for(int i(0); i != aSize; ++i)
        { 
          //Voxel* aVoxel(theVacantSpecies->getCompVoxel(i));
          unsigned aMol(theVacantSpecies->getCompMol(i));
          //if(aVoxel->id == theID)
          if(theIDs[aMol] == theID)
            {
              ++theMolSize;
            }
        }
      if(theMolSize > theMols.size())
        {
          theMols.resize(theMolSize);
        }
      theVariable->setValue(theMolSize);
      */
    }
  void setTagID(unsigned aBox, unsigned anIndex, unsigned anID)
    {
      theTags[aBox][anIndex].id = anID;
    }
  unsigned getTagID(unsigned aBox, unsigned anIndex)
    {
      return theTags[aBox][anIndex].id;
    }
  Tag& getTag(unsigned aBox, unsigned anIndex)
    {
      if(isTagged && anIndex != theMols[aBox].size())
        {
          return theTags[aBox][anIndex];
        }
      return theNullTag;
    }
  void addMol(unsigned aBox, unsigned aMol)
    {
      addMol(aBox, aMol, theNullTag);
    }
  Species* getMultiscaleVacantSpecies()
    {
      return theMultiscaleVacantSpecies;
    }
  void addMol(unsigned aBox, unsigned aMol, Tag& aTag)
    {
      if(isMultiscale)
        {
          Species* aSpecies(theStepper->id2species(theIDs[aBox][aMol]));
          if(aSpecies->getVacantSpecies() != theMultiscaleVacantSpecies)
            {
              doAddMol(aBox, aMol, aTag);
              addMultiscaleMol(aMol);
            }
        }
      else if(!isVacant)
        {
          doAddMol(aBox, aMol, aTag);
        }
      theIDs[aBox][aMol] = theID;
    }
  void doAddMol(unsigned aBox, unsigned aMol, Tag& aTag)
    {
      theMols[aBox].push_back(aMol);
      if(isTagged)
        {
          //If it is theNullTag:
          if(aTag.origin == theNullMol)
            {
              Tag aNewTag = {getMol(aBox, theMols[aBox].size()-1), theNullID};
              theTags[aBox].push_back(aNewTag);
            }
          else
            {
              theTags[aBox].push_back(aTag);
            }
        }
      theVariable->setValue(size());
    }
  void addMultiscaleMol(unsigned aMol)
    {
      /*
      unsigned coordA(aMol-vacStartMol);
      for(unsigned i(0); i != theIntersectLipids[coordA].size(); ++i)
        {
          unsigned coordB(theIntersectLipids[coordA][i]+lipStartMol);
          if(theIDs[coordB] == theMultiscaleVacantSpecies->getID())
            {
              theIDs[coordB] = theID;
            }
          else
            {
              multiscaleBind(coordB);
            }
        }
        */
    }
  void removeMultiscaleMol(unsigned aMol)
    {
      /*
      unsigned coordA(aMol-vacStartMol);
      for(unsigned i(0); i != theIntersectLipids[coordA].size(); ++i)
        {
          unsigned coordB(theIntersectLipids[coordA][i]+lipStartMol);
          if(theIDs[coordB] == theID)
            {
              theIDs[coordB] = theMultiscaleVacantSpecies->getID();
            }
          else
            {
              multiscaleUnbind(coordB);
            }
        }
        */
    }
  bool isIntersectMultiscale(unsigned aMol)
    {
      /*
      unsigned coordA(aMol-vacStartMol);
      for(unsigned i(0); i != theIntersectLipids[coordA].size(); ++i)
        {
          unsigned coordB(theIntersectLipids[coordA][i]+lipStartMol);
          unsigned anID(theIDs[coordB]);
          if(anID == theID ||
             std::find(theMultiscaleIDs.begin(), theMultiscaleIDs.end(),
                       anID) != theMultiscaleIDs.end())
            {
              return true;
            }
        }
        */
      return false;
    }
  bool isIntersectMultiscale(unsigned srcMol, unsigned tarMol)
    {
      bool isIntersect(false);
      /*
      unsigned coordA(srcMol-vacStartMol);
      std::vector<unsigned> temp;
      temp.resize(theIntersectLipids[coordA].size());
      for(unsigned i(0); i != theIntersectLipids[coordA].size(); ++i)
        {
          unsigned coordB(theIntersectLipids[coordA][i]+lipStartMol);
          temp[i] = theIDs[coordB];
          theIDs[coordB] = theSpeciesSize;
        }
      isIntersect = isIntersectMultiscale(tarMol);
      coordA = srcMol-vacStartMol;
      for(unsigned i(0); i != theIntersectLipids[coordA].size(); ++i)
        {
          unsigned coordB(theIntersectLipids[coordA][i]+lipStartMol);
          theIDs[coordB] = temp[i];
        }
        */
      return isIntersect;
    }
  void multiscaleBind(unsigned aMol)
    {
      /*
      unsigned anID(theIDs[aMol]);
      Species* source(theStepper->id2species(anID));
      Species* target(theStepper->id2species(theMultiscaleBindIDs[anID]));
      source->softRemoveMol(aMol);
      target->addMol(aMol);
      */
    }
  void multiscaleUnbind(unsigned aMol)
    {
      /*
      unsigned anID(theIDs[aMol]);
      Species* source(theStepper->id2species(anID));
      Species* target(theStepper->id2species(theMultiscaleUnbindIDs[anID]));
      source->softRemoveMol(aMol);
      target->addMol(aMol);
      */
    }
  void addCompVoxel(unsigned aBox, unsigned aMol)
    {
      theIDs[aBox][aMol] = theID;
      (*theCompMols)[aBox].push_back(aMol);
      //TODO: synchronize for multip
      theVariable->addValue(1);
    }
  String getIDString(Species* aSpecies)
    {
      Variable* aVariable(aSpecies->getVariable());
      if(aVariable)
        {
          return "["+aVariable->getSystemPath().asString()+":"+
            aVariable->getID()+"]["+int2str(aSpecies->getID())+"]";
        }
      else if(aSpecies->getID() == theNullID)
        {
          return "[theNullID]["+int2str(aSpecies->getID())+"]";
        }
      return "[unknown]";
    }
  String getIDString()
    {
      Variable* aVariable(getVariable());
      if(aVariable)
        {
          return "["+aVariable->getSystemPath().asString()+":"+
            aVariable->getID()+"]["+int2str(getID())+"]";
        }
      else if(getID() == theNullID)
        {
          return "[theNullID]["+int2str(getID())+"]";
        }
      return "[unknown]";
    }
  /*
  unsigned compVoxelSize()
    {
      //return theCompVoxels->size();
      return (*theCompMols->size();
    }
  unsigned getCompMol(unsigned aBox, unsigned index)
    {
      return (*theCompMols)[aBox][index];
    }
  Voxel* getCompVoxel(unsigned index)
    {
      return (*theCompVoxels)[index];
    }
    */
  unsigned getMolIndexFast(unsigned aBox, unsigned aMol)
    {
      for(unsigned i(0); i < theMols[aBox].size(); ++i)
        {
          if(theMols[aBox][i] == aMol)
            {
              return i;
            }
        }
      return theMols[aBox].size();
    }
  unsigned getMolIndex(unsigned aBox, unsigned aMol)
    {
      unsigned index(getMolIndexFast(aBox, aMol));
      if(index == theMols[aBox].size())
        { 
          if(isDiffusiveVacant || isReactiveVacant)
            {
              updateVacantMols();
            }
          index = getMolIndexFast(aBox, aMol);
          if(index == theMols[aBox].size())
            { 
              std::cout << "error in getting the index:" << getIDString() <<
               " size:" << theMols[aBox].size() << std::endl;
              return 0;
            }
        }
      return index;
    }
  //it is soft remove because the id of the molecule is not changed:
  void softRemoveMol(unsigned aMol)
    {
      /*
      if(isMultiscale)
        {
          Species* aSpecies(theStepper->id2species(theIDs[aMol]));
          if(aSpecies->getVacantSpecies() != theMultiscaleVacantSpecies)
            {
              softRemoveMolIndex(getMolIndex(aMol));
            }
        }
      else if(!isVacant)
        {
          softRemoveMolIndex(getMolIndex(aMol));
        }
        */
    }
  void removeMol(unsigned aMol)
    {
      /*
      if(isMultiscale)
        {
          Species* aSpecies(theStepper->id2species(theIDs[aMol]));
          if(aSpecies->getVacantSpecies() != theMultiscaleVacantSpecies)
            {
              removeMolIndex(getMolIndex(aMol));
            }
        }
      if(!isVacant)
        {
          removeMolIndex(getMolIndex(aMol));
        }
        */
    }
  void removeMolIndex(const unsigned aBox, const unsigned anIndex)
    {
      if(!isVacant)
        {
          theIDs[aBox][theMols[aBox][anIndex]] = theVacantID;
          softRemoveMolIndex(aBox, anIndex);
        }
    }
  void softRemoveMolIndex(const unsigned aBox, const unsigned anIndex)
    {
      if(isMultiscale)
        {
          /*
          removeMultiscaleMol(theMols[anIndex]);
          */
        }
      if(!isVacant)
        {
          theMols[aBox][anIndex] = theMols[aBox].back();
          theMols[aBox].pop_back();
          if(theMols[aBox].size()+1 == theLastMolSize[aBox])
            {
              theTars[aBox][anIndex] = theTars[aBox].back();
              theTars[aBox].pop_back();
              --theLastMolSize[aBox];
              --theCnt[aBox];
            }
          if(isTagged)
            {
              theTags[aBox][anIndex] = theTags[aBox].back();
              theTags.pop_back();
            }
          //theVariable->setValue(size());
          return;
        }
    }
  //Used to remove all molecules and free memory used to store the molecules
  void clearMols()
    {
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          theMols[i].resize(0);
        }
      theVariable->setValue(0);
    }
  //Used by the SpatiocyteStepper when resetting an interation, so must
  //clear the whole compartment using theComp->vacantSpecies->getVacantID():
  void removeMols()
    {
      /*
      if(!isCompVacant)
        {
          for(unsigned i(0); i < theMolSize; ++i)
            {
              theIDs[theMols[i]] = theVacantSpecies->getID();
            }
          theMolSize = 0;
          theVariable->setValue(theMolSize);
        }
        */
    }
  int getPopulateMolSize(unsigned aBox)
    {
      return theInitMolSize[aBox]-theMols[aBox].size();
    }
  int getTotalPopulateMolSize()
    {
      unsigned aSize(0);
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          aSize += getPopulateMolSize(i);
        }
      return aSize;
    }
  int getInitMolSize(unsigned aBox)
    {
      return theInitMolSize[aBox];
    }
  void initMolOrigins()
    {
      /*
      theMolOrigins.resize(theMolSize);
      for(unsigned i(0); i < theMolSize; ++i)
        {
          Origin& anOrigin(theMolOrigins[i]);
          anOrigin.point = theStepper->coord2point(getMol(i));
          anOrigin.row = 0;
          anOrigin.layer = 0;
          anOrigin.col = 0;
        }
        */
    }
  void removeBoundaryMols()
    {
      /*
      for(unsigned i(0); i < theMolSize; ++i)
        {
          if(theStepper->isBoundaryMol(getMol(i), theDimension))
            {
              std::cout << "is still there" << std::endl;
            }
        }
      theVariable->setValue(theMolSize);
      */
    }
  void relocateBoundaryMols()
    {
      /*
      for(unsigned i(0); i < theMolSize; ++i)
        {
          Origin anOrigin(theMolOrigins[i]);
          unsigned periodicMol(theStepper->getPeriodicMol(
                                                getMol(i),
                                                theDimension, &anOrigin));
          if(theIDs[periodicMol] == theVacantID)
            {
              theIDs[theMols[i]] = theVacantID;
              theMols[i] = periodicMol;
              theIDs[theMols[i]] = theID;
              theMolOrigins[i] = anOrigin;
            }
        }
        */
    }
  int getVacantID() const
    {
      return theVacantID;
    }
  Species* getVacantSpecies()
    {
      return theVacantSpecies;
    }
  void setVacantSpecies(Species* aVacantSpecies)
    {
      theVacantSpecies = aVacantSpecies;
      theVacantID = aVacantSpecies->getID();
    }
   const std::vector<double>& getBendAngles() const
    {
      return theBendAngles;
    }
  const Point getWestPoint() const
    {
      Point aWestPoint(theComp->centerPoint);
      aWestPoint.x = theComp->centerPoint.x-theComp->lengthX/2+
        theComp->lengthY/2;
      return aWestPoint;
    }
  const Point getEastPoint() const
    {
      Point anEastPoint(theComp->centerPoint); 
      anEastPoint.x = theComp->centerPoint.x+theComp->lengthX/2-
        theComp->lengthY/2;
      return anEastPoint;
    }
  double getCompRadius() const
    {
      return theComp->lengthY/2;
    }
  double getMolRadius() const
    {
      return theMolRadius;
    }
  double getDiffuseRadius() const
    {
      return theDiffuseRadius;
    }
  void setMolRadius(double aRadius)
    {
      theMolRadius = aRadius;
      theDiffuseRadius = aRadius;
    }
  void setDiffuseRadius(double aRadius)
    {
      theDiffuseRadius = aRadius;
    }
  Species* getDiffusionInfluencedReactantPair()
    {
      if(theDiffusionInfluencedReactantPairs.empty())
        {
          return NULL;
        }
      return theDiffusionInfluencedReactantPairs[0];
    }
  double getReactionProbability(int anID)
    {
      return theReactionProbabilities[anID];
    }
  double getMaxReactionProbability()
    {
      double maxProbability(0);
      for(std::vector<double>::const_iterator i(theReactionProbabilities.begin()); 
          i != theReactionProbabilities.end(); ++i)
        {
          if(maxProbability < *i)
            {
              maxProbability = *i;
            }
        }
      return maxProbability;
    }
  void rescaleReactionProbabilities(double aWalkProbability)
    {
      theWalkProbability = aWalkProbability;
      for(std::vector<double>::iterator i(theReactionProbabilities.begin()); 
          i != theReactionProbabilities.end(); ++i)
        {
          *i = (*i)*aWalkProbability;
        }
    }
  void setDiffusionInterval(double anInterval)
    {
      if(anInterval < theDiffusionInterval)
        {
          theDiffusionInterval = anInterval;
        }
      for(unsigned i(0); i != theTagSpeciesList.size(); ++i)
        {
          theTagSpeciesList[i]->setDiffusionInterval(theDiffusionInterval);
        }
    }
  unsigned getRandomIndex(unsigned aBox)
    {
      return gsl_rng_uniform_int(theRng, theMols[aBox].size());
    }
  unsigned getRandomMol(unsigned aBox)
    {
      return theMols[aBox][getRandomIndex(aBox)];
    }
  void addInterruptedProcess(SpatiocyteNextReactionProcess* aProcess)
    {
      if(std::find(theInterruptedProcesses.begin(),
                   theInterruptedProcesses.end(), aProcess) == 
         theInterruptedProcesses.end())
        {
          theInterruptedProcesses.push_back(aProcess);
        }
    }
  int getBendIndex(double aBendAngle)
    {
      for(unsigned i(0); i != theBendAngles.size(); ++i)
        {
          if(theBendAngles[i] == aBendAngle)
            {
              return i;
            }
        }
      return 0;
    }
  unsigned getRandomAdjoin(unsigned srcMol, int searchVacant)
    {
      std::vector<unsigned> compMols;
      /*
      if(searchVacant)
        { 
          for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
            {
              unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
              if(isPopulatable(aMol))
                {
                  compMols.push_back(aMol);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
            {
              unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
              if(theStepper->id2Comp(theIDs[aMol]) == theComp)
                {
                  compMols.push_back(aMol);
                }
            }
        }
        */
      return getRandomVacantMol(compMols);
    } 
  unsigned getBindingSiteAdjoin(unsigned srcMol, int bindingSite)
    {
      /*
      if(bindingSite < theInfo[srcMol].adjoinSize)
        { 
          unsigned aMol(theAdjoins[srcMol*theAdjoinSize+bindingSite]);
          if(isPopulatable(aMol))
            {
              return aMol;
            }
        }
        */
      return theNullMol;
    } 
  unsigned getRandomAdjoin(unsigned srcMol, Species* aTargetSpecies,
                           int searchVacant)
    {
      std::vector<unsigned> compMols;
      /*
      if(searchVacant)
        { 
          for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
            {
              unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
              if(theIDs[aMol] == aTargetSpecies->getID())
                {
                  compMols.push_back(aMol);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
            {
              unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
              if(theStepper->id2Comp(theIDs[aMol]) == theComp)
                {
                  compMols.push_back(aMol);
                }
            }
        }
        */
      return getRandomVacantMol(compMols, aTargetSpecies);
    } 
  unsigned getRandomAdjoin(unsigned srcMol, unsigned tarMol,
                           int searchVacant)
    {
      std::vector<unsigned> compMols;
      /*
      if(searchVacant)
        { 
          for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
            {
              unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
              if(aMol != tarMol && isPopulatable(aMol))
                {
                  compMols.push_back(aMol);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
            {
              unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
              if(theStepper->id2Comp(theIDs[aMol]) == theComp &&
                 aMol != tarMol)
                {
                  compMols.push_back(aMol);
                }
            }
        }
        */
      return getRandomVacantMol(compMols);
    }
  unsigned getAdjoinMolCnt(unsigned srcMol, Species* aTargetSpecies)
    {
      unsigned cnt(0);
      /*
      for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
        {
          if(theIDs[theAdjoins[srcMol*theAdjoinSize+i]] == aTargetSpecies->getID())
            {
              ++cnt;
            }
        }
        */
      return cnt;
    }
  unsigned getRandomAdjoin(unsigned srcMol, unsigned targetA,
                           unsigned targetB, int searchVacant)
    {
      std::vector<unsigned> compMols;
      /*
      if(srcMol != targetA && srcMol != targetB)
        {
          if(searchVacant)
            { 
              for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
                {
                  unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
                  if(isPopulatable(aMol))
                    {
                      compMols.push_back(aMol);
                    }
                }
            }
          else
            {
              for(unsigned i(0); i != theInfo[srcMol].adjoinSize; ++i)
                {
                  unsigned aMol(theAdjoins[srcMol*theAdjoinSize+i]);
                  if(theStepper->id2Comp(theIDs[aMol]) == theComp)
                    {
                      compMols.push_back(aMol);
                    }
                }
            }
        }
        */
      return getRandomVacantMol(compMols);
    }
  unsigned getRandomVacantMol(std::vector<unsigned>& aMols)
    {
      if(aMols.size())
        {
          const int r(gsl_rng_uniform_int(theRng, aMols.size())); 
          unsigned aMol(aMols[r]);
          if(isPopulatable(aMol))
            {
              return aMol;
            }
        }
      return theNullMol;
    }
  unsigned getRandomVacantMol(std::vector<unsigned>& aMols,
                                Species* aVacantSpecies)
    {
      /*
      if(aMols.size())
        {
          const int r(gsl_rng_uniform_int(theRng, aMols.size())); 
          unsigned aMol(aMols[r]);
          if(theIDs[aMol] == aVacantSpecies->getID())
            {
              return aMol;
            }
        }
        */
      return theNullMol;
    }
  unsigned getRandomCompMol(int searchVacant)
    {
      /*
      Species* aVacantSpecies(theComp->vacantSpecies);
      int aSize(aVacantSpecies->compVoxelSize());
      int r(gsl_rng_uniform_int(theRng, aSize));
      if(searchVacant)
        {
          for(int i(r); i != aSize; ++i)
            {
              unsigned aMol(aVacantSpecies->getCompMol(i));
              if(isPopulatable(aMol))
                {
                  return aMol;
                }
            }
          for(int i(0); i != r; ++i)
            {
              unsigned aMol(aVacantSpecies->getCompMol(i));
              if(isPopulatable(aMol))
                {
                  return aMol;
                }
            }
        }
      else
        {
          unsigned aMol(aVacantSpecies->getCompMol(r));
          if(isPopulatable(aMol))
            {
              return aMol;
            }
        }
        */
      return theNullMol;
    }
      /*
  unsigned getRandomAdjoinCompMol(Comp* aComp, int searchVacant)
    {
      int aSize(theVacantSpecies->size());
      int r(gsl_rng_uniform_int(theRng, aSize)); 
      return getRandomAdjoin(theVacantSpecies->getMol(r), searchVacant);
    }
    */
  void setVacStartMol(unsigned aMol)
    {
      vacStartMol = aMol;
    }
  void setLipStartMol(unsigned aMol)
    {
      lipStartMol = aMol;
    }
  void setIntersectLipids(Species* aLipid)
    {
      /*
      //Traverse through the entire compartment voxels:
      unsigned endA(vacStartMol+theVacantSpecies->size());
      unsigned endB(lipStartMol+aLipid->size());
      double dist((aLipid->getMolRadius()+theMolRadius)/
                  (2*theVoxelRadius));
      theIntersectLipids.resize(theVacantSpecies->size());
      for(unsigned i(vacStartMol); i != endA; ++i)
        {
          Point& pointA(theInfo[i].point);
          for(unsigned j(lipStartMol); j != endB; ++j)
            {
              Point& pointB(theInfo[j].point);
              if(getDistance(pointA, pointB) < dist)
                {
                  //We save j-lipStartMol and not the absolute coord
                  //since the absolute coord may change after resize 
                  //of lattice:
                  theIntersectLipids[i-vacStartMol
                    ].push_back(j-lipStartMol);
                }
            }
        }
        */
    }
  void setMultiscaleBindUnbindIDs(unsigned anID, unsigned aPairID)
    {
      if(std::find(theMultiscaleIDs.begin(), theMultiscaleIDs.end(),
                   anID) == theMultiscaleIDs.end())
        {
          theMultiscaleIDs.push_back(anID);
        }
      theMultiscaleBindIDs[aPairID] = anID;
      theMultiscaleUnbindIDs[anID] = aPairID;
    }
  //Get the fraction of number of nanoscopic molecules (anID) within the
  //multiscale molecule (index):
  double getMultiscaleBoundFraction(unsigned index, unsigned anID)
    {
      double fraction(0);
      /*
      if(isMultiscale)
        {
          unsigned i(getMol(index)-vacStartMol);
          for(unsigned j(0); j != theIntersectLipids[i].size(); ++j)
            {
              unsigned aMol(theIntersectLipids[i][j]+lipStartMol);
              if(theIDs[aMol] == anID)
                {
                  fraction += 1;
                }
            }
          fraction /= theIntersectLipids[i].size();
        }
        */
      return fraction;
    }
  unsigned getPopulatableSize()
    {
      if(isMultiscale)
        {
          /*
          if(!getIsPopulated())
            {
              std::cout << "The multiscale species:" << 
                getVariable()->getFullID().asString() << " has not yet " <<
                "been populated, but it being populated on." << std::endl;
            }
          thePopulatableMols.resize(0);
          for(unsigned i(0); i != theMolSize; ++i)
            {
              unsigned j(getMol(i)-vacStartMol);
              for(unsigned k(0); k != theIntersectLipids[j].size(); ++k)
                {
                  unsigned aMol(theIntersectLipids[j][k]+lipStartMol);
                  if(theIDs[aMol] == theID)
                    {
                      thePopulatableMols.push_back(aMol);
                    }
                }
            }
          return thePopulatableMols.size();
          */
        }
      return size();
    }
  unsigned getRandomPopulatableMol(unsigned aBox)
    {
      unsigned aMol;
      if(isMultiscale)
        {
          /*
          unsigned index(0);
          do
            {
              index = gsl_rng_uniform_int(theRng, thePopulatableMols.size());
            }
          while(theIDs[thePopulatableMols[index]] != theID);
          aMol =  thePopulatableMols[index];
          */
        }
      else
        {
          aMol = getRandomMol(aBox);
          while(theIDs[aBox][aMol] != theID)
            {
              aMol = getRandomMol(aBox);
            }
        }
      return aMol;
    }
  unsigned getPopulatableMol(unsigned aBox, unsigned index)
    {
      /*
      if(isMultiscale)
        {
          return thePopulatableMols[index];
        }
        */
      return getMol(aBox, index);
    }
  void setMultiscaleVacantSpecies(Species* aSpecies)
    {
      theMultiscaleVacantSpecies = aSpecies;
    }
  //Can aVoxel be populated by this species:
  bool isPopulatable(unsigned aMol)
    {
      /*
      if(isMultiscale)
        {
          if(isIntersectMultiscale(aMol))
            {
              return false;
            }
        }
      else if(theIDs[aMol] != theVacantID)
        {
          return false;
        }
        */
      return true;
    }
  //Can aVoxel of this species replaced by aSpecies:
  bool isReplaceable(unsigned aMol, Species* aSpecies)
    {
      if(getComp() != aSpecies->getComp() &&
         theID != aSpecies->getVacantID())
        {
          return false;
        }
      if(aSpecies->getIsMultiscale())
        {
          if(aSpecies->isIntersectMultiscale(aMol))
            {
              return false;
            }
        }
      return true;
    }
private:
  bool isCentered;
  bool isCompVacant;
  bool isDiffusing;
  bool isDiffusiveVacant;
  bool isFixedAdjoins;
  bool isGaussianPopulation;
  bool isInContact;
  bool isMultiscale;
  bool isOffLattice;
  bool isPolymer;
  bool isReactiveVacant;
  bool isSubunitInitialized;
  bool isTag;
  bool isTagged;
  bool isToggled;
  bool isVacant;
  const unsigned short theID;
  unsigned lipStartMol;
  unsigned theAdjoinSize;
  unsigned theBoxMaxSize;
  unsigned theBoxSize;
  unsigned theCollision;
  unsigned theDimension;
  unsigned theNullMol;
  unsigned theNullID;
  unsigned theSpeciesSize;
  unsigned vacStartMol;
  int thePolymerDirectionality;
  unsigned short theVacantID;
  double D;
  double theDiffuseRadius;
  double theDiffusionInterval;
  double theMolRadius;
  double theVoxelRadius;
  double theWalkProbability;
  const gsl_rng* theRng;
  Species* theVacantSpecies;
  Species* theMultiscaleVacantSpecies;
  Comp* theComp;
  MoleculePopulateProcessInterface* thePopulateProcess;
  SpatiocyteStepper* theStepper;
  Variable* theVariable;
  Tag theNullTag;
  std::vector<unsigned> theCnt;
  std::vector<unsigned> theInitMolSize;
  std::vector<unsigned> theLastMolSize;
  std::vector<bool> theFinalizeReactions;
  std::vector<unsigned> collisionCnts;
  std::vector<unsigned> theMultiscaleBindIDs;
  std::vector<unsigned> theMultiscaleIDs;
  std::vector<unsigned> theMultiscaleUnbindIDs;
  std::vector<unsigned> thePopulatableMols;
  std::vector<std::vector<Tag> > theTags;
  std::vector<double> theBendAngles;
  std::vector<double> theReactionProbabilities;
  std::vector<std::vector<unsigned> >* theCompMols;
  std::vector<Species*> theDiffusionInfluencedReactantPairs;
  std::vector<Species*> theTaggedSpeciesList;
  std::vector<Species*> theTagSpeciesList;
  std::vector<DiffusionInfluencedReactionProcess*> 
    theDiffusionInfluencedReactions;
  std::vector<SpatiocyteNextReactionProcess*> theInterruptedProcesses;
  std::vector<Origin> theMolOrigins;
  std::vector<std::vector<unsigned> > theMols;
  std::vector<std::vector<unsigned> > theTars;
  std::vector<std::vector<unsigned> > theRands;
  std::vector<std::vector<std::vector<std::vector<unsigned> > > > theAdjMols;
  std::vector<std::vector<std::vector<std::vector<unsigned> > > > theAdjTars;
  std::vector<std::vector<std::vector<std::vector<unsigned> > > > theAdjAdjMols;
  std::vector<std::vector<std::vector<std::vector<unsigned> > > > theAdjAdjTars;
  std::vector<std::vector<std::vector<std::vector<unsigned> > > > theBorderMols;
  std::vector<std::vector<std::vector<std::vector<unsigned> > > > theBorderTars;
  std::vector<std::vector<std::vector<unsigned> > > theRepeatAdjMols;
  std::vector<std::vector<std::vector<unsigned> > > theRepeatAdjTars;
  std::vector<std::vector<unsigned> >& theAdjoins;
  std::vector<std::vector<unsigned> >& theAdjBoxes;
  std::vector<std::vector<unsigned> >& theAdjAdjBoxes;
  std::vector<std::vector<VoxelInfo> >& theInfo;
  std::vector<std::vector<unsigned short> >& theIDs;
  std::vector<std::vector<unsigned> > theIntersectLipids;
  RandomLib::Random rng;          // Create r
  std::vector<Thread*> theThreads;
  unsigned theThreadSize;
};

#endif /* __Species_hpp */

