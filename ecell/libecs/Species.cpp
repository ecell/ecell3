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

#include "Species.hpp"

//all writing -> local
//all reading -> shared

void Species::updateBoxMols(const unsigned currBox, const unsigned r,
                            std::vector<unsigned>& aMols,
                            std::vector<unsigned>& aTars,
                            const std::vector<unsigned>& anAdjBoxes)
{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]/(theBoxSize/theThreadSize));
      const unsigned aBoxID(anAdjBoxes[i]%(theBoxSize/theThreadSize));
      //reading border mols, so shared:
      std::vector<unsigned>& borderMols(theThreads[adjBox
                                        ]->getBorderMols(currBox, r, aBoxID));
      std::vector<unsigned>& borderTars(theThreads[adjBox
                                        ]->getBorderTars(currBox, r, aBoxID));
      for(unsigned j(0); j < borderMols.size(); ++j)
        {
          aMols.push_back(borderMols[j]);
          aTars.push_back(borderTars[j]);
        }
      borderMols.resize(0);
      borderTars.resize(0);
    }
}

void Species::walkMols(std::vector<unsigned>& aMols,
                       const std::vector<unsigned>& aTars,
                       std::vector<unsigned short>& anIDs)
{
  for(unsigned i(0); i < aMols.size(); ++i)
    {
      const unsigned aTar(aTars[i]);
      const unsigned aTarMol(aTar%theBoxMaxSize);
      if(anIDs[aTarMol] == theVacantID)
        { 
          anIDs[aTarMol] = theID;
          anIDs[aMols[i]] = theVacantID;
          aMols[i] = aTarMol;
        }
    }
}

void Species::updateAdjMols(const unsigned currBox, const unsigned r,
                            std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                            std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                            const std::vector<unsigned>& anAdjBoxes)
{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]/(theBoxSize/theThreadSize));
      const unsigned aBoxID(anAdjBoxes[i]%(theBoxSize/theThreadSize));
      std::vector<unsigned>& repeatAdjMols(aRepeatAdjMols[anAdjBoxes[i]]);
      std::vector<unsigned>& repeatAdjTars(aRepeatAdjTars[anAdjBoxes[i]]);
      std::vector<unsigned>& adjMols(theThreads[adjBox
                                     ]->getAdjMols(currBox, r, aBoxID));
      std::vector<unsigned>& adjTars(theThreads[adjBox
                                     ]->getAdjTars(currBox, r, aBoxID));
      for(unsigned j(0); j != repeatAdjMols.size(); ++j)
        {
          adjMols.push_back(repeatAdjMols[j]);
          adjTars.push_back(repeatAdjTars[j]);
        }
      repeatAdjMols.resize(0);
      repeatAdjTars.resize(0);
    }
}

void Species::updateAdjAdjMols(const unsigned currBox, const unsigned r,
                               const std::vector<unsigned>& anAdjAdjBoxes)
{
  //for(unsigned i(0); i != theBoxSize; ++i)
  for(unsigned i(0); i != anAdjAdjBoxes.size(); ++i)
    {
      const unsigned adjAdjBox(anAdjAdjBoxes[i]/(theBoxSize/theThreadSize));
      const unsigned aBoxID(anAdjAdjBoxes[i]%(theBoxSize/theThreadSize));
      //const unsigned adjAdjBox(i);
      std::vector<unsigned>& adjAdjMols(theThreads[adjAdjBox
                                        ]->getAdjAdjMols(currBox, r, aBoxID));
      std::vector<unsigned>& adjAdjTars(theThreads[adjAdjBox
                                        ]->getAdjAdjTars(currBox, r, aBoxID));
      for(unsigned j(0); j != adjAdjMols.size(); ++j)
        {
          const unsigned aMol(adjAdjMols[j]);
          const unsigned aBox(aMol/theBoxMaxSize);
          theThreads[aBox/(theBoxSize/theThreadSize)]->pushAdj(currBox, r,
       aMol-theBoxMaxSize*aBox, adjAdjTars[j], aBox%(theBoxSize/theThreadSize));
        }
      adjAdjMols.resize(0);
      adjAdjTars.resize(0);
    }
}

void Species::walkAdjMols(const unsigned currBox, const unsigned r,
                          std::vector<unsigned>& aMols,
                          std::vector<unsigned short>& anIDs,
                          const std::vector<unsigned>& anAdjBoxes)
{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned aBox(anAdjBoxes[i]/(theBoxSize/theThreadSize));
      const unsigned aBoxID(anAdjBoxes[i]%(theBoxSize/theThreadSize));
      std::vector<unsigned>& adjMols(theThreads[aBox]->getAdjMols(currBox, r,
                                                                  aBoxID));
      std::vector<unsigned>& adjTars(theThreads[aBox]->getAdjTars(currBox, r,
                                                                  aBoxID));
      for(unsigned j(0); j < adjMols.size(); ++j)
        {
          const unsigned aTar(adjTars[j]);
          const unsigned aTarMol(aTar%theBoxMaxSize);
          if(anIDs[aTarMol] == theVacantID)
            {
              anIDs[aTarMol] = theID;
              theThreads[aBox]->setMolID(adjMols[j], theVacantID, aBoxID);
              //theIDs[aBox][adjMols[j]] = theVacantID;
              aMols.push_back(aTarMol);
              adjMols[j] = adjMols.back();
              adjMols.pop_back();
              adjTars[j] = adjTars.back();
              adjTars.pop_back();
              --j;
            }
        }
      adjTars.resize(0);
    }
}

void Species::setAdjTars(const unsigned currBox, const unsigned r,
                std::vector<std::vector<unsigned> >& aBorderMols,
                std::vector<std::vector<unsigned> >& aBorderTars,
                std::vector<std::vector<unsigned> >& anAdjAdjMols,
                std::vector<std::vector<unsigned> >& anAdjAdjTars,
                std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                const std::vector<unsigned>& anAdjBoxes,
                unsigned startRand,
                std::vector<unsigned>& aRands)

{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]);
      const unsigned aBoxID(anAdjBoxes[i]%(theBoxSize/theThreadSize));
      //reading adjMols, so get it from the thread:
      std::vector<unsigned>& adjMols(theThreads[anAdjBoxes[i]/
               (theBoxSize/theThreadSize)]->getAdjMols(currBox, r, aBoxID));
      const std::vector<unsigned>& anAdjoins(theAdjoins[adjBox]);
      for(unsigned j(0); j < adjMols.size(); ++j)
        {
          const unsigned aMol(adjMols[j]);
          const unsigned aTar(anAdjoins[aMol*theAdjoinSize+aRands[startRand++]]);
          const unsigned aBox(aTar/theBoxMaxSize);
          if(aBox == currBox) 
            {
              aRepeatAdjMols[adjBox].push_back(aMol);
              aRepeatAdjTars[adjBox].push_back(aTar);
            }
          else if(aBox == adjBox)
            {
              aBorderMols[adjBox].push_back(aMol);
              aBorderTars[adjBox].push_back(aTar);
            }
          else
            {
              anAdjAdjMols[aBox].push_back(theBoxMaxSize*adjBox+aMol);
              anAdjAdjTars[aBox].push_back(aTar);
            }
        }
      adjMols.resize(0);
    }
}

void Species::setRands(const unsigned currBox,
                       const unsigned r,
                       unsigned aSize,
                       const std::vector<unsigned>& anAdjBoxes,
                       std::vector<unsigned>& aRands,
                       RandomLib::Random& aRng)

{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]/(theBoxSize/theThreadSize));
      const unsigned aBoxID(anAdjBoxes[i]%(theBoxSize/theThreadSize));
      aSize += theThreads[adjBox]->getAdjMolsSize(currBox, r, aBoxID);
    }
  aRands.resize(aSize);
  for(unsigned i(0); i < aSize; ++i)
    {
      aRands[i] = aRng.IntegerC(11);
    }
}

void Species::setTars(const unsigned currBox,
                      std::vector<unsigned>& aMols,
                      std::vector<unsigned>& aTars,
                      std::vector<std::vector<unsigned> >& anAdjMols,
                      std::vector<std::vector<unsigned> >& anAdjTars,
                      const std::vector<unsigned>& anAdjoins,
                      std::vector<unsigned>& aRands)
{
  const unsigned aSize(aMols.size());
  aTars.resize(aSize);
  for(unsigned i(0); i < aSize; ++i)
    {
      aTars[i] = anAdjoins[aMols[i]*theAdjoinSize+aRands[i]];
    }
  for(unsigned i(0); i < aMols.size(); ++i)
    {
      if(aTars[i]/theBoxMaxSize != currBox)
        {
          anAdjMols[aTars[i]/theBoxMaxSize].push_back(aMols[i]);
          anAdjTars[aTars[i]/theBoxMaxSize].push_back(aTars[i]);
          aMols[i] = aMols.back();
          aMols.pop_back();
          aTars[i] = aTars.back();
          aTars.pop_back();
          --i;
        }
    }
}

void Species::walk(const unsigned anID, unsigned r, unsigned w,
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
           std::vector<unsigned>& aRands)
{
  updateBoxMols(anID, r, aMols, aTars, anAdjBoxes);
  walkMols(aMols, aTars, anIDs);
  updateAdjMols(anID, r, aRepeatAdjMols, aRepeatAdjTars, anAdjBoxes);
  updateAdjAdjMols(anID, r, anAdjAdjBoxes); 
  walkAdjMols(anID, r, aMols, anIDs, anAdjBoxes);
  setRands(anID, r, aMols.size(), anAdjBoxes, aRands, aRng);
  setTars(anID, aMols, aTars, anAdjMols[w], anAdjTars[w], anAdjoins, aRands);
  setAdjTars(anID, r, aBorderMols[w], aBorderTars[w], anAdjAdjMols[w], anAdjAdjTars[w], aRepeatAdjMols, aRepeatAdjTars, anAdjBoxes, aMols.size(), aRands);
  /*
  if(anID == 1)
    {
      std::cout << "m:" << aMols.size() << " t:" << aTars.size() << " bM0:" << theThreads[anID]->getBorderMolsSize(0) << " bM1:" << theThreads[anID]->getBorderMolsSize(1) << " bT0:" << theThreads[anID]->getBorderTarsSize(0) << " bT1:" << theThreads[anID]->getBorderTarsSize(1) << " aM0:" << theThreads[anID]->getAdjMolsSize(0) << " aM1:" << theThreads[anID]->getAdjMolsSize(1) << " aT0:" << theThreads[anID]->getAdjTarsSize(0) << " aT1:" << theThreads[anID]->getAdjTarsSize(1) <<  " aaM0:" << theThreads[anID]->getAdjAdjMolsSize(0) << " aaM1:" << theThreads[anID]->getAdjAdjMolsSize(1) << " aaT0:" << theThreads[anID]->getAdjAdjTarsSize(0) << " aaT1:" << theThreads[anID]->getAdjAdjTarsSize(1) << " raM:" << theThreads[anID]->getRepeatAdjMolsSize() << " raT:" << theThreads[anID]->getRepeatAdjTarsSize() << std::endl;
    }
    */


  /*
  updateBoxMols(theBorderMols[r][anID], theBorderTars[r][anID], theMols[anID],
                theTars[anID], theAdjBoxes[anID]);
  walkMols(theMols[anID], theTars[anID], theIDs[anID]);
  updateAdjMols(theRepeatAdjMols[anID], theRepeatAdjTars[anID],
                theAdjMols[r][anID], theAdjTars[r][anID], theAdjBoxes[anID]);
  updateAdjAdjMols(theAdjAdjMols[r][anID], theAdjAdjTars[r][anID],
                theAdjMols[r][anID], theAdjTars[r][anID]);
  walkAdjMols(theMols[anID], theAdjMols[r][anID], theAdjTars[r][anID],
              theIDs[anID], theAdjBoxes[anID]);
  setAdjTars(theBorderMols[w], theBorderTars[w], theAdjAdjMols[w],
             theAdjAdjTars[w], theRepeatAdjMols[anID], theRepeatAdjTars[anID],
             theAdjMols[r][anID], theAdjBoxes[anID], anID);
  setTars(theMols[anID], theTars[anID], theAdjMols[w], theAdjTars[w], anID,
          theAdjoins[anID]);
          */
}

void Species::updateMols()
{
  if(isDiffusiveVacant || isReactiveVacant)
    {
      updateVacantMols();
    }
  else if(isTag)
    {
      updateTagMols();
    }
  if(!theID && theStepper->getCurrentTime() > 0)
    {
      for(unsigned i(0); i != theBoxSize; ++i)
        {
          const unsigned aThreadID(i/(theBoxSize/theThreadSize));
          const unsigned aBoxID(i%(theBoxSize/theThreadSize));
          theThreads[aThreadID]->updateMols(theMols[i], aBoxID);
        }
    }
}


/*
void Species::setTars(std::vector<unsigned>& aMols,
           std::vector<unsigned>& aTars,
           std::vector<std::vector<std::vector<unsigned> > >& aNextMols,
           std::vector<std::vector<std::vector<unsigned> > >& aNextTars,
           const unsigned currBox,
           const std::vector<unsigned>& anAdjoins)
{
  aTars.resize(0);
  for(unsigned i(0); i < aMols.size(); ++i)
    {
      unsigned& aMol(aMols[i]);
      const unsigned aTar(anAdjoins[aMol*theAdjoinSize+rng.IntegerC(11)]);
      if(aTar/theBoxMaxSize == currBox) 
        {
          aTars.push_back(aTar);
        }
      else
        {
          aNextMols[aTar/theBoxMaxSize][currBox].push_back(aMol);
          aNextTars[aTar/theBoxMaxSize][currBox].push_back(aTar);
          aMol = aMols.back();
          aMols.pop_back();
          --i;
        }
    }
}
void Species::walkMols(std::vector<unsigned>& aMols,
            const std::vector<unsigned>& aTars,
            std::vector<unsigned short>& anIDs)
{
  for(unsigned i(0); i < aMols.size(); ++i)
    {
      const unsigned aTar(aTars[i]);
      const unsigned aTarMol(aTar%theBoxMaxSize);
      if(anIDs[aTarMol] == theVacantID)
        {
          if(theWalkProbability == 1 ||
             gsl_rng_uniform(theRng) < theWalkProbability)
            {
              anIDs[aTarMol] = theID;
              anIDs[aMols[i]] = theVacantID;
              aMols[i] = aTarMol;
            }
        }
    }
}
void Species::walkAdjMols(std::vector<unsigned>& aMols,
               std::vector<std::vector<unsigned> >& aBoxAdjMols,
               std::vector<std::vector<unsigned> >& aBoxAdjTars, 
               std::vector<unsigned short>& anIDs,
               const std::vector<unsigned>& anAdjBoxes)
{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned aBox(anAdjBoxes[i]);
      std::vector<unsigned>& adjMols(aBoxAdjMols[aBox]);
      std::vector<unsigned>& adjTars(aBoxAdjTars[aBox]);
      for(unsigned j(0); j < adjMols.size(); ++j)
        {
          const unsigned aTar(adjTars[j]);
          const unsigned aTarMol(aTar%theBoxMaxSize);
          if(anIDs[aTarMol] == theVacantID)
            {
              anIDs[aTarMol] = theID;
              theIDs[aBox][adjMols[j]] = theVacantID;
              aMols.push_back(aTarMol);
              adjMols[j] = adjMols.back();
              adjMols.pop_back();
              adjTars[j] = adjTars.back();
              adjTars.pop_back();
              --j;
            }
        }
      adjTars.resize(0);
    }
}
void Species::setAdjTars(std::vector<std::vector<std::vector<unsigned> > >&
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
              const unsigned currBox)

{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]);
      const std::vector<unsigned>& anAdjoins(theAdjoins[adjBox]);
      std::vector<unsigned>& adjMols(anAdjMols[adjBox]);
      for(unsigned j(0); j < adjMols.size(); ++j)
        {
          const unsigned aMol(adjMols[j]);
          const unsigned aTar(anAdjoins[aMol*theAdjoinSize+
                              rng.IntegerC(11)]);
          const unsigned aBox(aTar/theBoxMaxSize);
          if(aBox == currBox) 
            {
              aRepeatAdjMols[adjBox].push_back(aMol);
              aRepeatAdjTars[adjBox].push_back(aTar);
            }
          else if(aBox == adjBox)
            {
              aBorderMols[adjBox][currBox].push_back(aMol);
              aBorderTars[adjBox][currBox].push_back(aTar);
            }
          else
            {
              anAdjAdjMols[aBox][currBox].push_back(theBoxMaxSize*adjBox+
                                                     aMol);
              anAdjAdjTars[aBox][currBox].push_back(aTar);
            }
        }
      adjMols.resize(0);
    }
}

void Species::updateBoxMols(std::vector<std::vector<unsigned> >& aBorderMols,
                 std::vector<std::vector<unsigned> >& aBorderTars,
                 std::vector<unsigned>& aMols,
                 std::vector<unsigned>& aTars,
                 const std::vector<unsigned>& anAdjBoxes)
{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]);
      std::vector<unsigned>& borderMols(aBorderMols[adjBox]);
      std::vector<unsigned>& borderTars(aBorderTars[adjBox]);
      for(unsigned j(0); j < borderMols.size(); ++j)
        {
          aMols.push_back(borderMols[j]);
          aTars.push_back(borderTars[j]);
        }
      borderMols.resize(0);
      borderTars.resize(0);
    }
}
void Species::updateAdjMols(std::vector<std::vector<unsigned> >& aRepeatAdjMols,
                 std::vector<std::vector<unsigned> >& aRepeatAdjTars,
                 std::vector<std::vector<unsigned> >& anAdjMols,
                 std::vector<std::vector<unsigned> >& anAdjTars,
                 const std::vector<unsigned>& anAdjBoxes)
{
  for(unsigned i(0); i != anAdjBoxes.size(); ++i)
    {
      const unsigned adjBox(anAdjBoxes[i]);
      std::vector<unsigned>& repeatAdjMols(aRepeatAdjMols[adjBox]);
      std::vector<unsigned>& repeatAdjTars(aRepeatAdjTars[adjBox]);
      std::vector<unsigned>& adjMols(anAdjMols[adjBox]);
      std::vector<unsigned>& adjTars(anAdjTars[adjBox]);
      for(unsigned j(0); j != repeatAdjMols.size(); ++j)
        {
          adjMols.push_back(repeatAdjMols[j]);
          adjTars.push_back(repeatAdjTars[j]);
        }
      repeatAdjMols.resize(0);
      repeatAdjTars.resize(0);
    }
}
void Species::updateAdjAdjMols(std::vector<std::vector<unsigned> >& anAdjAdjMols,
                    std::vector<std::vector<unsigned> >& anAdjAdjTars,
                    std::vector<std::vector<unsigned> >& anAdjMols,
                    std::vector<std::vector<unsigned> >& anAdjTars)

{
  //for(unsigned i(0); i != anAdjAdjBoxes.size(); ++i)
  for(unsigned i(0); i != theBoxSize; ++i)
    {
      //const unsigned adjAdjBox(anAdjAdjBoxes[i]);
      const unsigned adjAdjBox(i);
      std::vector<unsigned>& adjAdjMols(anAdjAdjMols[adjAdjBox]);
      std::vector<unsigned>& adjAdjTars(anAdjAdjTars[adjAdjBox]);
      for(unsigned j(0); j != adjAdjMols.size(); ++j)
        {
          const unsigned aMol(adjAdjMols[j]);
          const unsigned aBox(aMol/theBoxMaxSize);
          anAdjMols[aBox].push_back(aMol-theBoxMaxSize*aBox);
          anAdjTars[aBox].push_back(adjAdjTars[j]);
        }
      adjAdjMols.resize(0);
      adjAdjTars.resize(0);
    }
}
*/
