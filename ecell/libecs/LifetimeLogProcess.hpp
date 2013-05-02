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


#ifndef __LifetimeLogProcess_hpp
#define __LifetimeLogProcess_hpp

#include <IteratingLogProcess.hpp>
#include <ReactionProcess.hpp>

LIBECS_DM_CLASS(LifetimeLogProcess, IteratingLogProcess)
{ 
public:
  LIBECS_DM_OBJECT(LifetimeLogProcess, Process)
    {
      INHERIT_PROPERTIES(IteratingLogProcess);
    }
  LifetimeLogProcess():
    konCnt(0)
  {
    FileName = "LifetimeLog.csv";
  }
  virtual ~LifetimeLogProcess() {}
  virtual void initialize();
  virtual void initializeFirst();
  virtual void initializeSecond();
  virtual void initializeFifth();
  virtual void initializeLastOnce();
  virtual void fire();
  virtual void interruptedPre(ReactionProcess*);
  virtual void interruptedPost(ReactionProcess*);
  virtual bool isDependentOnPre(const ReactionProcess*);
  virtual bool isDependentOnPost(const ReactionProcess*);
private:
  bool isInVariableReferences(const VariableReferenceVector&, const int,
                              const Variable*) const;
  void logTrackedMolecule(ReactionProcess*, Species*, const Voxel*);
  void initTrackedMolecule(Species*);
private:
  unsigned konCnt;
  std::vector<bool> isTrackedSpecies;
  std::vector<bool> isUntrackedSpecies;
  std::vector<bool> isBindingSite;
  std::vector<unsigned> availableTagIDs;
  std::vector<double> theTagTimes;
};

#endif /* __LifetimeLogProcess_hpp */
