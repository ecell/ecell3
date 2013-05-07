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


#ifndef __SPATIOCYTEPROCESSINTERFACE_HPP
#define __SPATIOCYTEPROCESSINTERFACE_HPP

#include <RandomLib/Random.hpp>
#include <SpatiocyteCommon.hpp>

class ReactionProcess;

class SpatiocyteProcessInterface
{ 
public:
  virtual ~SpatiocyteProcessInterface() {}
  virtual void initializeFirst() = 0;
  virtual void initializeSecond() = 0;
  virtual void initializeThird() = 0;
  virtual void initializeFourth() = 0;
  virtual void initializeFifth() = 0;
  virtual void initializeLastOnce() = 0;
  virtual void printParameters() = 0;
  virtual void setPriorityQueue(ProcessPriorityQueue*) = 0;
  virtual void setTime(double) = 0;
  virtual double getTime() const = 0;
  virtual int getPriority() const = 0;
  virtual void setQueueID(ProcessID) = 0;
  virtual void addSubstrateInterrupt(Species*, Voxel*) = 0;
  virtual void removeSubstrateInterrupt(Species*, Voxel*) = 0;
  virtual void setLatticeProperties(std::vector<Voxel>*, unsigned,
                                    unsigned, unsigned, RandomLib::Random*) = 0;
  virtual void substrateValueChanged(double) = 0;
  virtual void fire() = 0;
  virtual unsigned getLatticeResizeCoord(unsigned) = 0;
  virtual void updateResizedLattice() = 0;
  virtual bool getIsPriorityQueued() = 0;
  virtual bool getIsExternInterrupted() = 0;
  virtual double getNewInterval() = 0;
  virtual double getInitInterval() = 0;
  virtual String getIDString() = 0;
  virtual void finalizeFire() = 0;
  virtual void interruptedPre(ReactionProcess*) = 0;
  virtual void interruptedPost(ReactionProcess*) = 0;
  virtual bool isDependentOn(const Process*) const = 0;
  virtual bool isDependentOnPre(const ReactionProcess*) = 0;
  virtual bool isDependentOnPost(const ReactionProcess*) = 0;
};

#endif /* __SPATIOCYTEPROCESSINTERFACE_HPP */
