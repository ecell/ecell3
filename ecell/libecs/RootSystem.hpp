//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#ifndef ___ROOTSYSTEM_H___
#define ___ROOTSYSTEM_H___
#include "System.h"
#include "Primitive.h"


class RootSystem : public System
{

public:

  class MalformedSystemName : public NotFound
    {
    public:
      MalformedSystemName( StringCref method, StringCref message ) 
	: NotFound( method, message ) {}
      const String what() const { return "Malformed system name."; }
    };

public:

  RootSystem();
  ~RootSystem();

  int check();

  SystemPtr getSystem( SystemPathCref systempath )
    throw( NotFound, MalformedSystemName );
  Primitive getPrimitive( FQPNCref fqpn ) 
    throw( InvalidPrimitiveType, NotFound );

  virtual void initialize();

  StepperLeaderRef    getStepperLeader()    const { return theStepperLeader; }

  ReactorMakerRef     reactorMaker()     const { return theReactorMaker; }
  SubstanceMakerRef   substanceMaker()   const { return theSubstanceMaker; }
  SystemMakerRef      systemMaker()      const { return theSystemMaker; }
  StepperMakerRef     stepperMaker()     const { return theStepperMaker; }
  AccumulatorMakerRef accumulatorMaker() const { return theAccumulatorMaker; }

  virtual const char* const className() const { return "RootSystem"; }

private:

  void install();

private:

  StepperLeaderRef theStepperLeader;

  ReactorMakerRef theReactorMaker;
  SubstanceMakerRef theSubstanceMaker;
  SystemMakerRef theSystemMaker;
  StepperMakerRef theStepperMaker;
  AccumulatorMakerRef theAccumulatorMaker;

};

extern RootSystemPtr theRootSystem;

#endif /* ___ROOTSYSTEM_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
