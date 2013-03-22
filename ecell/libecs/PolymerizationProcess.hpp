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


#ifndef __PolymerizationProcess_hpp
#define __PolymerizationProcess_hpp

#include "DiffusionInfluencedReactionProcess.hpp"
#include "PolymerFragmentationProcess.hpp"
#include "SpatiocytePolymer.hpp"

LIBECS_DM_CLASS(PolymerizationProcess, DiffusionInfluencedReactionProcess)
{ 
public:
  LIBECS_DM_OBJECT(PolymerizationProcess, Process)
    {
      INHERIT_PROPERTIES(DiffusionInfluencedReactionProcess);
      PROPERTYSLOT_SET_GET(Real, BendAngle);
      PROPERTYSLOT_SET_GET(Real, CylinderYaw);
      PROPERTYSLOT_SET_GET(Real, SphereYaw);
    }
  SIMPLE_SET_GET_METHOD(Real, BendAngle);
  SIMPLE_SET_GET_METHOD(Real, CylinderYaw);
  SIMPLE_SET_GET_METHOD(Real, SphereYaw);
  PolymerizationProcess():
    BendAngle(0),
    CylinderYaw(0),
    theMonomerLength(1),
    SphereYaw(0) {}
  virtual ~PolymerizationProcess() {}
  virtual void initialize()
    {
      DiffusionInfluencedReactionProcess::initialize();
    }
  virtual void initializeFourth();
  virtual bool react(unsigned int, unsigned int);
  virtual bool isInterrupting(Process*);
  virtual void finalizeReaction();
  void resetSubunit(Subunit*);
  void removeContPoint(Subunit*,  Point*);
protected:
  void initSubunits(Species*);
  void initSubunit(unsigned int, Species*);
  void initJoinSubunit(Voxel*, Species*, Subunit*);
  void addContPoint(Subunit*,  Point*);
  void updateSharedLipidsID(Voxel*);
  void removeLipid(Subunit*, unsigned int);
  double setImmediateTargetVoxel(Subunit*, unsigned int);
  bool setExtendedTargetVoxel(Subunit*, unsigned int, double);
  int setExistingTargetVoxel(Subunit*, unsigned int);
  bool setTargetVoxels(Subunit*);
  Voxel* getTargetVoxel(Subunit*);
  void removeUnboundTargetVoxels(Subunit*);
  Voxel* setNewTargetVoxel(Subunit*, unsigned int);
  unsigned int sortNearestTargetVoxels(Subunit*, unsigned int, std::vector<Voxel*>&,
                                       std::vector<double>&, std::vector<int>&);
  int getLocation(double x)
    {
      if(x > theMinX && x < theMaxX)
        {
          return CYLINDER;
        }
      return ELLIPSOID;
    }
  virtual void pushNewBend(Subunit*, double);
  virtual void pushJoinBend(Subunit*, Subunit*, unsigned int);
  virtual Bend* getNewReverseBend(Point*, Point*, Bend*);
  virtual Bend* getReverseBend(Bend*);
  virtual Point getNextPoint(Point*, Bend*); 
  virtual void getCylinderDcm(double, double, double*);
  virtual void getSphereDcm(double, double, double*);
  virtual void initSphereDcm();
  virtual void pinStep(double*, double*, double*, double*);
protected:
  unsigned int theBendIndexA;
  unsigned int theBendIndexB;
  double BendAngle;
  double CylinderYaw;
  double theInitSphereDcm[9];
  double theMaxX;
  double theMinX;
  double theMonomerLength;
  double theOriY;
  double theOriZ;
  double theRadius;
  double SphereYaw;
};

#endif /* __PolymerizationProcess_hpp */
