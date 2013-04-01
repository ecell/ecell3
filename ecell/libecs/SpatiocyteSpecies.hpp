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


#ifndef __SpatiocyteSpecies_hpp
#define __SpatiocyteSpecies_hpp

#include <sstream>
#include <libecs/Variable.hpp>
#include <gsl/gsl_randist.h>
#include "SpatiocyteCommon.hpp"
#include "SpatiocyteStepper.hpp"
#include "SpatiocyteProcessInterface.hpp"
#include "DiffusionInfluencedReactionProcess.hpp"
#include "MoleculePopulateProcessInterface.hpp"


namespace libecs
{

using namespace libecs;

// The size of Coord must be 128 bytes to avoid cacheline splits
// The Core 2 has 64-byte cacheline
static double getDistance(Point* aSourcePoint, Point* aDestPoint)
{
  return sqrt(pow(aDestPoint->x-aSourcePoint->x, 2)+
              pow(aDestPoint->y-aSourcePoint->y, 2)+
              pow(aDestPoint->z-aSourcePoint->z, 2));
}


/*
 * isVacant: vacant species definition: a species on which other species can
 * diffuse on or occupy. One major optimization in speed for vacant species is
 * that theMolecules list is not updated when other species diffuse on it. 
 * There are four possible types of vacant species:
 * 1. !isDiffusiveVacant && !isReactiveVacant: a vacant species
 *    whose theMolecules list and theMoleculeSize are never updated. This is
 *    the most basic version. This vacant species never diffuses, or reacts
 *    using SNRP. 
 * 2. !isDiffusiveVacant && isReactiveVacant: a vacant species which is a
 *    substrate of a SNRP reaction. In this case theMoleculeSize is updated
 *    before the step interval of SNRP is calculated, and theMolecules list is
 *    updated before the SNRP reaction (fire) is executed to get a valid
 *    molecule list. Set by SNRP.
 * 3. isDiffusiveVacant && !isReactiveVacant: a vacant species which also
 *    diffuses. In this case, theMolecules list and theMoleculeSize are
 *    updated just before it is diffused. Set by DiffusionProcess.   
 * 4. isDiffusiveVacant && isReactiveVacant: a vacant species which reacts
 *    using SNRP and also diffuses.
 * 5. isCompVacant: the VACANT species declared for each compartment. It can
 *    also be isDiffusiveVacant and isReactiveVacant. Set during compartment
 *    registration. It also persistently stores all the compartment voxels.
 *    Referred to as theVacantSpecies. For isCompVacant, initially there
 *    are no molecules in its list. All voxels are stored in theCompVoxels. Only
 *    if it is updated before being called by VisualizationLogProcess, SNRP or
 *    DiffusionProcess, it will be theMolecules will be populated with the
 *    CompCoords.
 * 6. isVacant {isCompVacant; isDiffusiveVacant; isReactiveVacant): the
 *    general name used to identify either isCompVacant, isDiffusiveVacant or
 *    isReactiveVacant to reduce comparison operations.
 */

class Species
{
public:
  Species(SpatiocyteStepper* aStepper, Variable* aVariable, int anID, 
          int anInitCoordSize, const gsl_rng* aRng, double voxelRadius,
          std::vector<Voxel>& aLattice):
    isCentered(false),
    isCompVacant(false),
    isDiffusing(false),
    isDiffusiveVacant(false),
    isFixedAdjoins(false),
    isGaussianPopulation(false),
    isInContact(false),
    isOffLattice(false),
    isPolymer(false),
    isReactiveVacant(false),
    isSubunitInitialized(false),
    isTag(false),
    isTagged(false),
    isVacant(false),
    theID(anID),
    theCollision(0),
    theInitCoordSize(anInitCoordSize),
    theMoleculeSize(0),
    D(0),
    theDiffusionInterval(libecs::INF),
    theWalkProbability(1),
    theRadius(voxelRadius),
    theRng(aRng),
    thePopulateProcess(NULL),
    theStepper(aStepper),
    theVariable(aVariable),
    theCompVoxels(&theMolecules),
    theLattice(aLattice) {}
  ~Species() {}
  void initialize(int speciesSize, int anAdjoiningCoordSize,
                  unsigned aNullCoord, unsigned aNullID)
    {
      theAdjoiningCoordSize = anAdjoiningCoordSize;
      theNullCoord = aNullCoord;
      theNullID = aNullID;
      theReactionProbabilities.resize(speciesSize);
      theDiffusionInfluencedReactions.resize(speciesSize);
      theFinalizeReactions.resize(speciesSize);
      for(int i(0); i != speciesSize; ++ i)
        {
          theDiffusionInfluencedReactions[i] = NULL;
          theReactionProbabilities[i] = 0;
          theFinalizeReactions[i] = false;
        }
      if(theComp)
        {
          setVacantSpecies(theComp->vacantSpecies);
        }
      theNullTag.origin = theNullCoord;
      theNullTag.id = theNullID;
    }

static String int2str(int anInt)
{
  std::stringstream aStream;
  aStream << anInt;
  return aStream.str();
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
      else if(theMoleculeSize)
        {
          std::cout << "Warning: Species " <<
            theVariable->getFullID().asString() <<
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
  void populateCompUniform(unsigned voxelIDs[], unsigned* aCount)
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformDense(this, voxelIDs, aCount);
        }
      else if(theMoleculeSize)
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not CoordPopulated." << std::endl;
        }
    }
  void populateCompUniformSparse()
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformSparse(this);
        }
      else if(theMoleculeSize)
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not CoordPopulated." << std::endl;
        }
    }
  void populateUniformOnDiffusiveVacant()
    {
      if(thePopulateProcess)
        {
          thePopulateProcess->populateUniformOnDiffusiveVacant(this);
        }
      else if(theMoleculeSize)
        {
          std::cout << "Species:" << theVariable->getFullID().asString() <<
            " not CoordPopulated." << std::endl;
        }
    }
  Variable* getVariable() const
    {
      return theVariable;
    }
  std::vector<unsigned> getSourceCoords()
    {
      std::vector<unsigned> aCoords;
      for(unsigned i(0); i != theMoleculeSize; ++i)
        {
          std::vector<unsigned>& 
            aSourceCoords(theMolecules[i]->subunit->sourceCoords);
          for(unsigned j(0); j != aSourceCoords.size(); ++j)
            {
              if(aSourceCoords[j] != theNullCoord)
                {
                  aCoords.push_back(aSourceCoords[j]);
                }
            }
        }
      return aCoords;
    }
  std::vector<unsigned> getTargetCoords()
    {
      std::vector<unsigned> aCoords;
      for(unsigned i(0); i != theMoleculeSize; ++i)
        {
          std::vector<unsigned>& 
            aTargetCoords(theMolecules[i]->subunit->targetCoords);
          for(unsigned j(0); j != aTargetCoords.size(); ++j)
            {
              if(aTargetCoords[j] != theNullCoord)
                {
                  aCoords.push_back(aTargetCoords[j]);
                }
            }
        }
      return aCoords;
    }
  std::vector<unsigned> getSharedCoords()
    {
      std::vector<unsigned> aCoords;
      for(unsigned i(0); i != theMoleculeSize; ++i)
        {
          std::vector<unsigned>& 
            aSharedLipids(theMolecules[i]->subunit->sharedLipids);
          for(unsigned j(0); j != aSharedLipids.size(); ++j)
            {
              if(aSharedLipids[j] != theNullCoord)
                {
                  aCoords.push_back(aSharedLipids[j]);
                }
            }
        }
      return aCoords;
    }
  unsigned size() const
    {
      return theMoleculeSize;
    }
  Voxel* getMolecule(int anIndex)
    {
      return theMolecules[anIndex];
    }
  Point getPoint(int anIndex)
    {
      if(isOffLattice)
        {
          if(theMolecules[anIndex]->point)
            {
              return *theMolecules[anIndex]->point;
            }
          return theStepper->coord2point(getCoord(anIndex));
        }
      else if(isPolymer)
        {
          return theMolecules[anIndex]->subunit->subunitPoint;
        }
      return theStepper->coord2point(getCoord(anIndex));
    }
  unsigned short getID() const
    {
      return theID;
    }
  double getMeanSquaredDisplacement()
    {
      if(!theMoleculeSize)
        {
          return 0;
        }
      double aDisplacement(0);
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          Point aCurrentPoint(theStepper->getPeriodicPoint(
                                                 getCoord(i),
                                                 theDimension,
                                                 &theMoleculeOrigins[i]));
          double aDistance(getDistance(&theMoleculeOrigins[i].point,
                                       &aCurrentPoint));
          aDisplacement += aDistance*aDistance;
        }
      return aDisplacement*pow(theRadius*2, 2)/theMoleculeSize;
    }
  void setCollision(unsigned aCollision)
    {
      theCollision = aCollision;
    }
  void setIsSubunitInitialized()
    {
      isSubunitInitialized = true;
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
      theInitCoordSize = theMoleculeSize;
      getVariable()->setValue(theMoleculeSize);
    }
  void finalizeSpecies()
    {
      if(theCollision)
        {
          collisionCnts.resize(theMoleculeSize);
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
          for(unsigned i(0); i != theComp->species.size(); ++i)
            {
              if(theComp->species[i]->getIsDiffusiveVacant())
                {
                  std::random_shuffle(theMolecules.begin(), theMolecules.end());
                  break;
                }
            }
          if(isTagged)
            {
              for(unsigned i(0); i != theMoleculeSize; ++i)
                {
                  theTags[i].origin = getCoord(i);
                }
            }
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
      return theMoleculeSize == theInitCoordSize;
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
    }
  void addCollision(Voxel* aVoxel)
    {
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          if(aVoxel == theMolecules[i])
            {
              ++collisionCnts[i];
              return;
            }
        }
      std::cout << "error in species add collision" << std::endl;
    }
  void walk()
    {
      unsigned beginMoleculeSize(theMoleculeSize);
      for(unsigned i(0); i < beginMoleculeSize && i < theMoleculeSize; ++i)
        {
          Voxel* source(theMolecules[i]);
          int size;
          if(isFixedAdjoins)
            {
              size = theAdjoiningCoordSize;
            }
          else
            {
              size = source->diffuseSize;
            }
          Voxel* target(&theLattice[source->adjoiningCoords[
                        gsl_rng_uniform_int(theRng, size)]]);
          if(target->id == theVacantID)
            {
              if(theWalkProbability == 1 ||
                 gsl_rng_uniform(theRng) < theWalkProbability)
                {
                  target->id = theID;
                  source->id = theVacantID;
                  theMolecules[i] = target;
                }
            }
          else if(theDiffusionInfluencedReactions[target->id])
            {
              //If it meets the reaction probability:
              if(gsl_rng_uniform(theRng) < theReactionProbabilities[target->id])
                { 
                  Species* targetSpecies(theStepper->id2species(target->id));
                  unsigned targetIndex(targetSpecies->getIndex(target));
                  if(theCollision)
                    { 
                      ++collisionCnts[i];
                      targetSpecies->addCollision(target);
                      if(theCollision != 2)
                        {
                          return;
                        }
                    }
                  unsigned aMoleculeSize(theMoleculeSize);
                  react(i, targetIndex, targetSpecies, target);
                  //Only rewalk the current pointed molecule if it is an
                  //unwalked molecule (not a product of the reaction):
                  if(theMoleculeSize < aMoleculeSize)
                    {
                      --i;
                    }
                }
            }
        }
    }
  void walkVacant()
    {
      updateVacantMolecules();
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          Voxel* source(theMolecules[i]);
          int size;
          if(isFixedAdjoins)
            {
              size = theAdjoiningCoordSize;
            }
          else
            {
              size = source->diffuseSize;
            }
          Voxel* target(&theLattice[source->adjoiningCoords[
                        gsl_rng_uniform_int(theRng, size)]]);
          if(target->id == theVacantID)
            {
              if(theWalkProbability == 1 ||
                 gsl_rng_uniform(theRng) < theWalkProbability)
                {
                  target->id = theID;
                  source->id = theVacantID;
                  theMolecules[i] = target;
                }
            }
        }
    }
  void react(unsigned sourceIndex, unsigned targetIndex,
             Species* targetSpecies, Voxel* target)
    {
      DiffusionInfluencedReactionProcess* aReaction(
               theDiffusionInfluencedReactions[targetSpecies->getID()]);
      unsigned indexA(sourceIndex);
      unsigned indexB(targetIndex);
      if(aReaction->getA() != this)
        {
          indexA = targetIndex; 
          indexB = sourceIndex;
        }
      if(aReaction->react(indexA, indexB))
        {
          //Soft remove the source molecule, i.e., keep the id:
          softRemoveMolecule(sourceIndex);
          //Soft remove the target molecule:
          //Make sure the targetIndex is valid:
          //Target and Source are same species:
          //For some reason if I use theMolecules[sourceIndex] instead
          //of getMolecule(sourceIndex) the walk method becomes
          //much slower when it is only diffusing without reacting:
          if(targetSpecies == this && getMolecule(sourceIndex) == target)
            {
              softRemoveMolecule(sourceIndex);
            }
          else
            {
              targetSpecies->softRemoveMolecule(targetIndex);
            }
          theFinalizeReactions[targetSpecies->getID()] = true;
        }
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
  unsigned getCoord(unsigned anIndex)
    {
      return theMolecules[anIndex]->coord;
    }
  void removeSurfaces()
    {
      int newCoordSize(0);
      for(unsigned i(0); i < theMoleculeSize; ++i) 
        {
          unsigned aCoord(getCoord(i));
          if(theStepper->isRemovableEdgeCoord(aCoord, theComp))
            {
              Comp* aSuperComp(
                 theStepper->system2Comp(theComp->system->getSuperSystem())); 
              aSuperComp->vacantSpecies->addCompVoxel(aCoord);
            }
          else 
            { 
              theMolecules[newCoordSize] = &theLattice[aCoord];
              ++newCoordSize; 
            }
        }
      theMoleculeSize = newCoordSize;
      //Must resize, otherwise compVoxelSize will be inaccurate:
      theMolecules.resize(theMoleculeSize);
      theVariable->setValue(theMoleculeSize);
    }
  void removePeriodicEdgeVoxels()
    {
      int newCoordSize(0);
      for(unsigned i(0); i < theMoleculeSize; ++i) 
        {
          unsigned aCoord(getCoord(i));
          if(theStepper->isPeriodicEdgeCoord(aCoord, theComp))
            {
              theLattice[aCoord].id = theLattice[theNullCoord].id;
            }
          else 
            { 
              theMolecules[newCoordSize] = &theLattice[aCoord];
              ++newCoordSize; 
            }
        }
      theMoleculeSize = newCoordSize;
      //Must resize, otherwise compVoxelSize will be inaccurate:
      theMolecules.resize(theMoleculeSize);
      theVariable->setValue(theMoleculeSize);
    }
  void updateSpecies()
    {
      if(isCompVacant && (isDiffusiveVacant || isReactiveVacant))
        {
          theCompVoxels = new std::vector<Voxel*>;
          for(unsigned i(0); i != theMoleculeSize; ++i)
            { 
              theCompVoxels->push_back(theMolecules[i]);
            }
        }
    }
  //If it isReactiveVacant it will only be called by SNRP when it is substrate
  //If it isDiffusiveVacant it will only be called by DiffusionProcess before
  //being diffused. So we need to only check if it isVacant:
  void updateMolecules()
    {
      if(isDiffusiveVacant || isReactiveVacant)
        {
          updateVacantMolecules();
        }
      else if(isTag)
        {
          updateTagMolecules();
        }
    }
  //If it isReactiveVacant it will only be called by SNRP when it is substrate:
  void updateMoleculeSize()
    {
      if(isDiffusiveVacant || isReactiveVacant)
        {
          updateVacantCoordSize();
        }
    }
  void updateTagMolecules()
    {
      theMoleculeSize = 0;
      for(unsigned i(0); i != theTaggedSpeciesList.size(); ++i)
        {
          Species* aSpecies(theTaggedSpeciesList[i]);
          for(unsigned j(0); j != aSpecies->size(); ++j)
            {
              if(aSpecies->getTagID(j) == theID)
                {
                  Voxel* aVoxel(aSpecies->getMolecule(j));
                  ++theMoleculeSize;
                  if(theMoleculeSize > theMolecules.size())
                    {
                      theMolecules.push_back(aVoxel);
                    }
                  else
                    {
                      theMolecules[theMoleculeSize-1] = aVoxel;
                    }
                }
            }
        }
    }
  //Even if it is a isCompVacant, this method will be called by
  //VisualizationLogProcess, or SNRP if it is Reactive, or DiffusionProcess
  //if it is Diffusive:
  void updateVacantMolecules()
    {
      theMoleculeSize = 0;
      int aSize(theVacantSpecies->compVoxelSize());
      for(int i(0); i != aSize; ++i)
        { 
          Voxel* aVoxel(theVacantSpecies->getCompVoxel(i));
          if(aVoxel->id == theID)
            {
              ++theMoleculeSize;
              if(theMoleculeSize > theMolecules.size())
                {
                  theMolecules.push_back(aVoxel);
                }
              else
                {
                  theMolecules[theMoleculeSize-1] = aVoxel;
                }
            }
        }
      theVariable->setValue(theMoleculeSize);
    }
  void updateVacantCoordSize()
    {
      theMoleculeSize = 0;
      int aSize(theVacantSpecies->compVoxelSize());
      for(int i(0); i != aSize; ++i)
        { 
          Voxel* aVoxel(theVacantSpecies->getCompVoxel(i));
          if(aVoxel->id == theID)
            {
              ++theMoleculeSize;
            }
        }
      if(theMoleculeSize > theMolecules.size())
        {
          theMolecules.resize(theMoleculeSize);
        }
      theVariable->setValue(theMoleculeSize);
    }
  void setTagID(unsigned anIndex, unsigned anID)
    {
      theTags[anIndex].id = anID;
    }
  unsigned getTagID(unsigned anIndex)
    {
      return theTags[anIndex].id;
    }
  Tag& getTag(unsigned anIndex)
    {
      if(isTagged)
        {
          return theTags[anIndex];
        }
      return theNullTag;
    }
  void addMolecule(Voxel* aVoxel, Tag& aTag)
    {
      aVoxel->id = theID;
      if(!isVacant)
        {
          ++theMoleculeSize; 
          if(theMoleculeSize > theMolecules.size())
            {
              theMolecules.resize(theMoleculeSize);
              theTags.resize(theMoleculeSize);
            }
          theMolecules[theMoleculeSize-1] = aVoxel;
          if(isTagged)
            {
              //If it is theNullTag:
              if(aTag.origin == theNullCoord)
                {
                  Tag aNewTag = {getCoord(theMoleculeSize-1), theNullID};
                  theTags[theMoleculeSize-1] = aNewTag;
                }
              else
                {
                  theTags[theMoleculeSize-1] = aTag;
                }
            }
          theVariable->setValue(theMoleculeSize);
        }
    }
  void addMolecule(Voxel* aVoxel)
    {
      addMolecule(aVoxel, theNullTag);
    }
  void addCompVoxel(unsigned aCoord)
    {

      theLattice[aCoord].id = theID;
      theCompVoxels->push_back(&theLattice[aCoord]);
      ++theMoleculeSize;
      theVariable->setValue(theMoleculeSize);
    }
  unsigned compVoxelSize()
    {
      return theCompVoxels->size();
    }
  Voxel* getCompVoxel(unsigned index)
    {
      return (*theCompVoxels)[index];
    }
  unsigned getIndexFast(Voxel* aVoxel)
    {
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          if(theMolecules[i] == aVoxel)
            {
              return i;
            }
        }
      return theMoleculeSize;
    }
  unsigned getIndex(Voxel* aVoxel)
    {
      unsigned index(getIndexFast(aVoxel));
      if(index == theMoleculeSize)
        { 
          if(isDiffusiveVacant || isReactiveVacant)
            {
              updateVacantMolecules();
            }
          index = getIndexFast(aVoxel);
          if(index == theMoleculeSize)
            { 
              std::cout << "error in getting the index:" << 
                getVariable()->getFullID().asString() << std::endl;
              return 0;
            }
        }
      return index;
    }
  //it is soft remove because the id of the molecule is not changed:
  void softRemoveMolecule(Voxel* aVoxel)
    {
      if(!isVacant)
        {
          softRemoveMolecule(getIndex(aVoxel));
        }
    }
  void removeMolecule(Voxel* aVoxel)
    {
      if(!isVacant)
        {
          removeMolecule(getIndex(aVoxel));
        }
    }
  void removeMolecule(unsigned anIndex)
    {
      if(!isVacant)
        {
          theMolecules[anIndex]->id = theVacantID;
          softRemoveMolecule(anIndex);
        }
    }
  void softRemoveMolecule(unsigned anIndex)
    {
      if(!isVacant)
        {
          theMolecules[anIndex] = theMolecules[--theMoleculeSize];
          if(isTagged)
            {
              theTags[anIndex] = theTags[theMoleculeSize];
            }
          theVariable->setValue(theMoleculeSize);
          return;
        }
    }
  //Used to remove all molecules and free memory used to store the molecules
  void clearMolecules()
    {
      theMolecules.resize(0);
      theMoleculeSize = 0;
      theVariable->setValue(0);
    }
  //Used by the SpatiocyteStepper when resetting an interation, so must
  //clear the whole compartment using theComp->vacantSpecies->getVacantID():
  void removeMolecules()
    {
      if(!isCompVacant)
        {
          for(unsigned i(0); i < theMoleculeSize; ++i)
            {
              theMolecules[i]->id = theVacantSpecies->getID();
            }
          theMoleculeSize = 0;
          theVariable->setValue(theMoleculeSize);
        }
    }
  int getPopulateCoordSize()
    {
      return theInitCoordSize-theMoleculeSize;
    }
  int getInitCoordSize()
    {
      return theInitCoordSize;
    }
  void setInitCoordSize(const unsigned& val)
    {
      theInitCoordSize = val;
    }
  void initMoleculeOrigins()
    {
      theMoleculeOrigins.resize(theMoleculeSize);
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          Origin& anOrigin(theMoleculeOrigins[i]);
          anOrigin.point = theStepper->coord2point(getCoord(i));
          anOrigin.row = 0;
          anOrigin.layer = 0;
          anOrigin.col = 0;
        }
    }
  void removeBoundaryMolecules()
    {
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          if(theStepper->isBoundaryCoord(getCoord(i), theDimension))
            {
              std::cout << "is still there" << std::endl;
            }
        }
      theVariable->setValue(theMoleculeSize);
    }
  void relocateBoundaryMolecules()
    {
      for(unsigned i(0); i < theMoleculeSize; ++i)
        {
          Origin anOrigin(theMoleculeOrigins[i]);
          unsigned periodicCoord(theStepper->getPeriodicCoord(
                                                getCoord(i),
                                                theDimension, &anOrigin));
          if(theLattice[periodicCoord].id == theVacantID)
            {
              theMolecules[i]->id = theVacantID;
              theMolecules[i] = &theLattice[periodicCoord];
              theMolecules[i]->id = theID;
              theMoleculeOrigins[i] = anOrigin;
            }
        }
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
  double getRadius() const
    {
      return theRadius;
    }
  void setRadius(double aRadius)
    {
      theRadius = aRadius;
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
  unsigned getRandomIndex()
    {
      return gsl_rng_uniform_int(theRng, theMoleculeSize);
    }
  Voxel* getRandomMolecule()
    {
      return theMolecules[getRandomIndex()];
    }
  void addInterruptedProcess(SpatiocyteProcessInterface* aProcess)
    {
      theInterruptedProcesses.push_back(aProcess);
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
  Voxel* getRandomAdjoiningVoxel(Voxel* source, int searchVacant)
    {
      std::vector<unsigned> compCoords;
      if(searchVacant)
        { 
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theLattice[aCoord].id == theVacantID)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theStepper->id2Comp(theLattice[aCoord].id) == theComp)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      return getRandomVacantVoxel(compCoords);
    } 
  Voxel* getBindingSiteAdjoiningVoxel(Voxel* source, int bindingSite)
    {
      if(bindingSite < source->adjoiningSize)
        { 
          unsigned aCoord(source->adjoiningCoords[bindingSite]);
          if(theLattice[aCoord].id == theVacantID)
            {
              return &theLattice[aCoord];
            }
        }
      return NULL;
    } 
  Voxel* getRandomAdjoiningVoxel(Voxel* source,
                                 Species* aTargetSpecies,
                                 int searchVacant)
    {
      std::vector<unsigned> compCoords;
      if(searchVacant)
        { 
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theLattice[aCoord].id == aTargetSpecies->getID())
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theStepper->id2Comp(theLattice[aCoord].id) == theComp)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      return getRandomVacantVoxel(compCoords, aTargetSpecies);
    } 
  Voxel* getRandomAdjoiningVoxel(Voxel* source, Voxel* target,
                                 int searchVacant)
    {
      std::vector<unsigned> compCoords;
      if(searchVacant)
        { 
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theLattice[aCoord].id == theVacantID && 
                 &theLattice[aCoord] != target)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theStepper->id2Comp(theLattice[aCoord].id) == theComp &&
                 &theLattice[aCoord] != target)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      return getRandomVacantVoxel(compCoords);
    }
  unsigned getAdjoiningMoleculeCnt(Voxel* source, Species* aTargetSpecies)
    {
      unsigned cnt(0);
      for(unsigned i(0); i != source->adjoiningSize; ++i)
        {
          if(theLattice[source->adjoiningCoords[i]].id == 
             aTargetSpecies->getID())
            {
              ++cnt;
            }
        }
      return cnt;
    }
  Voxel* getRandomAdjoiningVoxel(Voxel* source, Voxel* targetA, Voxel* targetB,
                                 int searchVacant)
    {
      std::vector<unsigned> compCoords;
      if(searchVacant)
        { 
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theLattice[aCoord].id == theVacantID &&
                 source != targetA && source != targetB)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      else
        {
          for(unsigned i(0); i != source->adjoiningSize; ++i)
            {
              unsigned aCoord(source->adjoiningCoords[i]);
              if(theStepper->id2Comp(theLattice[aCoord].id) == theComp &&
                 source != targetA && source != targetB)
                {
                  compCoords.push_back(aCoord);
                }
            }
        }
      return getRandomVacantVoxel(compCoords);
    }
  Voxel* getRandomVacantVoxel(std::vector<unsigned>& aCoords)
    {
      if(aCoords.size())
        {
          const int r(gsl_rng_uniform_int(theRng, aCoords.size())); 
          unsigned aCoord(aCoords[r]);
          if(theLattice[aCoord].id == theVacantID)
            {
              return &theLattice[aCoord];
            }
        }
      return NULL;
    }
  Voxel* getRandomVacantVoxel(std::vector<unsigned>& aCoords,
                              Species* aVacantSpecies)
    {
      if(aCoords.size())
        {
          const int r(gsl_rng_uniform_int(theRng, aCoords.size())); 
          unsigned aCoord(aCoords[r]);
          if(theLattice[aCoord].id == aVacantSpecies->getID())
            {
              return &theLattice[aCoord];
            }
        }
      return NULL;
    }
  Voxel* getRandomCompVoxel(int searchVacant)
    {
      Species* aVacantSpecies(theComp->vacantSpecies);
      int aSize(aVacantSpecies->compVoxelSize());
      int r(gsl_rng_uniform_int(theRng, aSize));
      if(searchVacant)
        {
          for(int i(r); i != aSize; ++i)
            {
              Voxel* aVoxel(aVacantSpecies->getCompVoxel(i));
              if(aVoxel->id == theVacantID)
                {
                  return aVoxel;
                }
            }
          for(int i(0); i != r; ++i)
            {
              Voxel* aVoxel(aVacantSpecies->getCompVoxel(i));
              if(aVoxel->id == theVacantID)
                {
                  return aVoxel;
                }
            }
        }
      else
        {
          Voxel* aVoxel(aVacantSpecies->getCompVoxel(r));
          if(aVoxel->id == theVacantID)
            {
              return aVoxel;
            }
        }
      return NULL;
    }
  Voxel* getRandomAdjoiningCompVoxel(Comp* aComp, int searchVacant)
    {
      int aSize(theVacantSpecies->size());
      int r(gsl_rng_uniform_int(theRng, aSize)); 
      Voxel* aVoxel(theVacantSpecies->getMolecule(r));
      return getRandomAdjoiningVoxel(aVoxel, searchVacant);
    }
  //We need to updateMolecules to set the valid address of voxels
  //since they may have been changed when theLattice is resized by 
  //processes:
  void updateMoleculePointers()
    {
      for(unsigned i(0); i != theCoords.size(); ++i)
        {
          theMolecules[i] = &theLattice[theCoords[i]];
        }
      theCoords.resize(0);
    }
  void saveCoords()
    {
      theCoords.resize(theMoleculeSize);
      for(unsigned i(0); i != theMoleculeSize; ++i)
        {
          theCoords[i] = getCoord(i);
        }
    }
private:
  bool isCentered;
  bool isCompVacant;
  bool isDiffusing;
  bool isDiffusiveVacant;
  bool isFixedAdjoins;
  bool isGaussianPopulation;
  bool isInContact;
  bool isOffLattice;
  bool isPolymer;
  bool isReactiveVacant;
  bool isSubunitInitialized;
  bool isTag;
  bool isTagged;
  bool isVacant;
  const unsigned short theID;
  unsigned theCollision;
  unsigned theDimension;
  unsigned theInitCoordSize;
  unsigned theMoleculeSize;
  unsigned theNullCoord;
  unsigned theNullID;
  unsigned theAdjoiningCoordSize;
  int thePolymerDirectionality;
  int theVacantID;
  double D;
  double theDiffusionInterval;
  double theWalkProbability;
  double theRadius;
  const gsl_rng* theRng;
  Species* theVacantSpecies;
  Comp* theComp;
  MoleculePopulateProcessInterface* thePopulateProcess;
  SpatiocyteStepper* theStepper;
  Variable* theVariable;
  Tag theNullTag;
  std::vector<bool> theFinalizeReactions;
  std::vector<unsigned> collisionCnts;
  std::vector<unsigned> theCoords;
  std::vector<Tag> theTags;
  std::vector<double> theBendAngles;
  std::vector<double> theReactionProbabilities;
  std::vector<Voxel*> theMolecules;
  std::vector<Voxel*>* theCompVoxels;
  std::vector<Species*> theDiffusionInfluencedReactantPairs;
  std::vector<Species*> theTaggedSpeciesList;
  std::vector<Species*> theTagSpeciesList;
  std::vector<DiffusionInfluencedReactionProcess*> 
    theDiffusionInfluencedReactions;
  std::vector<SpatiocyteProcessInterface*> theInterruptedProcesses;
  std::vector<Origin> theMoleculeOrigins;
  std::vector<Voxel>& theLattice;
};

}

#endif /* __SpatiocyteSpecies_hpp */
