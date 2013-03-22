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


#ifndef __HistogramLogProcess_hpp
#define __HistogramLogProcess_hpp

#include <fstream> //provides ofstream
#include <math.h>
#include "IteratingLogProcess.hpp"
#include "SpatiocyteSpecies.hpp"

LIBECS_DM_CLASS(HistogramLogProcess, IteratingLogProcess)
{ 
public:
  LIBECS_DM_OBJECT(HistogramLogProcess, Process)
    {
      INHERIT_PROPERTIES(IteratingLogProcess);
      PROPERTYSLOT_SET_GET(Integer, Bins);
      PROPERTYSLOT_SET_GET(Integer, Collision);
      PROPERTYSLOT_SET_GET(Real, Radius);
      PROPERTYSLOT_SET_GET(Real, Length);
      PROPERTYSLOT_SET_GET(Real, OriginX);
      PROPERTYSLOT_SET_GET(Real, OriginY);
      PROPERTYSLOT_SET_GET(Real, OriginZ);
      PROPERTYSLOT_SET_GET(Real, RotateX);
      PROPERTYSLOT_SET_GET(Real, RotateY);
      PROPERTYSLOT_SET_GET(Real, RotateZ);
    }
  SIMPLE_SET_GET_METHOD(Integer, Bins);
  SIMPLE_SET_GET_METHOD(Integer, Collision);
  SIMPLE_SET_GET_METHOD(Real, Radius);
  SIMPLE_SET_GET_METHOD(Real, Length);
  SIMPLE_SET_GET_METHOD(Real, OriginX);
  SIMPLE_SET_GET_METHOD(Real, OriginY);
  SIMPLE_SET_GET_METHOD(Real, OriginZ);
  SIMPLE_SET_GET_METHOD(Real, RotateX);
  SIMPLE_SET_GET_METHOD(Real, RotateY);
  SIMPLE_SET_GET_METHOD(Real, RotateZ);
  HistogramLogProcess():
    Bins(1),
    Collision(0),
    OriginX(0),
    OriginY(0),
    OriginZ(0),
    Radius(12.5e-9),
    RotateX(0),
    RotateY(0),
    RotateZ(0) 
  {
    FileName = "HistogramLog.csv";
    LogStart = 1e-8;
  }
  virtual ~HistogramLogProcess() {}
  virtual void initializeFifth();
  virtual void initializeLastOnce();
  virtual void fire();
  virtual void saveFile();
  virtual void saveBackup();
  virtual void logValues();
  virtual void logCollision();
  virtual void logDensity();
  virtual void initLogValues();
  virtual void logFile();
  virtual void saveFileHeader(std::ofstream&);
  void initializeVectors();
  bool isInside(unsigned int&, Point);
protected:
  unsigned int Bins;
  unsigned int Collision;
  double binInterval;
  double Length;
  double OriginX;
  double OriginY;
  double OriginZ;
  double Radius;
  double RotateX;
  double RotateY;
  double RotateZ;
  double VoxelDiameter;
  Point C;
  Point E;
  Point D;
  Comp* theComp;
  std::vector<std::vector<std::vector<double> > > theLogValues;
};

#endif /* __HistogramLogProcess_hpp */
