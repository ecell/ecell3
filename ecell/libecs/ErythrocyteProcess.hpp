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


#ifndef __ErythrocyteProcess_hpp
#define __ErythrocyteProcess_hpp

#include "SpatiocyteProcess.hpp"
#include "SpatiocyteSpecies.hpp"

LIBECS_DM_CLASS(ErythrocyteProcess, SpatiocyteProcess)
{
public:
  LIBECS_DM_OBJECT(ErythrocyteProcess, Process)
    {
      INHERIT_PROPERTIES(Process);
      PROPERTYSLOT_SET_GET(Real, EdgeLength);
    }
  ErythrocyteProcess():
    isCompartmentalized(false),
    EdgeLength(60e-9),
    theVertexSpecies(NULL),
    theSpectrinSpecies(NULL) {}
  virtual ~ErythrocyteProcess() {}
  SIMPLE_SET_GET_METHOD(Real, EdgeLength);
  virtual void initialize()
    {
      if(isInitialized)
        {
          return;
        }
      SpatiocyteProcess::initialize();
      for(VariableReferenceVector::iterator
          i(theSortedVariableReferences.begin());
          i != theSortedVariableReferences.end(); ++i)
        {
          Species* aSpecies(theSpatiocyteStepper->variable2species(
                                   (*i).getVariable())); 
          if(!theSpectrinSpecies)
            {
              theSpectrinSpecies = aSpecies;
            }
          else if(!theVertexSpecies)
            {
              theVertexSpecies = aSpecies;
            }
          else
            {
              THROW_EXCEPTION(ValueError, String(
                              getPropertyInterface().getClassName()) +
                              "[" + getFullID().asString() + 
                              "]: An ErythrocyteProcess requires only " +
                              "two variable references as the spectrin and " +
                              "vertex species of the Erythrocyte " + 
                              "compartment but " +
                              getIDString(theSpectrinSpecies) + ", " +
                              getIDString(theVertexSpecies) + " and " +
                              getIDString(aSpecies) + " are given."); 
            }
        }
      if(!theSpectrinSpecies)
        {
          THROW_EXCEPTION(ValueError, String(
                          getPropertyInterface().getClassName()) +
                          "[" + getFullID().asString() + 
                          "]: An ErythrocyteProcess requires one " +
                          "nonHD variable reference coefficient as the " +
                          "spectrin species, but none is given."); 
        }
      if(!theVertexSpecies)
        {
          THROW_EXCEPTION(ValueError, String(
                          getPropertyInterface().getClassName()) +
                          "[" + getFullID().asString() + 
                          "]: An ErythrocyteProcess requires one " +
                          "nonHD variable reference coefficient as the " +
                          "vertex species, but none is given."); 
        }
      VoxelDiameter = theSpatiocyteStepper->getVoxelRadius()*2;
      EdgeLength /= VoxelDiameter;
      TriangleAltitude = cos(M_PI/6)*EdgeLength;
    }
  virtual void initializeThird();
  void populateMolecules();
  void initializeDirectionVectors();
  void initializeProtofilaments();
  void normalize(Point&);
  bool isInsidePlane(Point&, Point&, Point&);
  bool isOnPlane(Point&, Point&, Point&, unsigned int);
  bool isOnLowerPlanes(Point&, unsigned int, Point&);
  bool isOnUpperPlanes(Point&, unsigned int, Point&);
  unsigned int getIntersectCount(Point&, unsigned int);
  void rotatePointAlongVector(Point&, Point&, Point&, double);
protected:
  bool isCompartmentalized;
  double EdgeLength;
  double VoxelDiameter;
  double TriangleAltitude;
  Point Y; //Direction vector along the rotated positive y-axis
  Point Z; //Direction vector along the rotated positive z-axis
  Point R; //Direction vector along rotated north east
  Point L; //Direction vector along rotated north west
  Point C; //Center point
  Species* theVertexSpecies;
  Species* theSpectrinSpecies;
  Comp* theComp;
  std::vector<unsigned int> filamentCoords;
};

#endif /* __ErythrocyteProcess_hpp */




