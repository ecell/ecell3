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

#include "Util.hpp"
#include "Stepper.hpp"
#include "System.hpp"


namespace libecs
{


  //FIXME: should be merged with PropertySlot::SlotTypes

  template <class T,typename Ret>
  class VoidObjectMethod
  {
    typedef Ret (T::* Method )( void ) const;

  public:
    VoidObjectMethod( T& anObject, Method aMethod )
      :
      theObject( anObject ),
      theMethod( aMethod )
    {
      ; // do nothing
    }

    const Ret operator()( void ) const 
    {
      return ( theObject.*theMethod )();
    }

  private:

    T&     theObject;
    Method theMethod;

  };

  typedef VoidObjectMethod<RootSystem,const Real> GetCurrentTimeMethodType;

  class RootSystem 
    : 
    public System
  {

  public:

    RootSystem();
    ~RootSystem();

    void makeSlots();


    virtual const SystemPath getSystemPath() const;
    virtual SystemPtr getSystem( StringCref id );
    //    virtual SystemPtr getSystem( SystemPathCref systempath );

    virtual void initialize();

    StepperLeaderRef    getStepperLeader()    { return theStepperLeader; }

    ReactorMakerRef     getReactorMaker()     { return theReactorMaker; }
    SubstanceMakerRef   getSubstanceMaker()   { return theSubstanceMaker; }
    SystemMakerRef      getSystemMaker()      { return theSystemMaker; }
    StepperMakerRef     getStepperMaker()     { return theStepperMaker; }
    AccumulatorMakerRef getAccumulatorMaker() { return theAccumulatorMaker; }

    const Real getCurrentTime() const;

    virtual StringLiteral getClassName() const { return "RootSystem"; }


  private:

    StepperLeader       theStepperLeader;

    ReactorMakerRef     theReactorMaker;
    SubstanceMakerRef   theSubstanceMaker;
    SystemMakerRef      theSystemMaker;
    StepperMakerRef     theStepperMaker;
    AccumulatorMakerRef theAccumulatorMaker;

  };


} // namespace libecs

#endif /* ___ROOTSYSTEM_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
