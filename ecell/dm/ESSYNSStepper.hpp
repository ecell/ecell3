//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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

#ifndef __ESSYNS_HPP
#define __ESSYNS_HPP


// #include <iostream>

#include "libecs/DifferentialStepper.hpp"
#include "ESSYNSProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ESSYNSStepper, AdaptiveDifferentialStepper )
{

 public:

    class Interpolant
      :
      public libecs::Interpolant
    {
    public:
      Interpolant( ESSYNSStepperRef aStepper, 
		     VariablePtr const aVariablePtr )
	:
	libecs::Interpolant( aVariablePtr ),
	theStepper( aStepper ),
	theIndex( theStepper.getVariableIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getDifference( const Real aTime, const Real anInterval )
      {
	// First order interpolation.  This should be overridden in
	// higher order DifferentialSteppers.
	return theStepper.getVelocityBuffer()[ theIndex ] * anInterval;
      }
      

    protected:

      ESSYNSStepperRef    theStepper;
      VariableVector::size_type theIndex;

    };

public:

  LIBECS_DM_OBJECT( ESSYNSStepper, Stepper )
    {
      INHERIT_PROPERTIES( AdaptiveDifferentialStepper );
      PROPERTYSLOT_SET_GET( Integer, TaylorOrder );
    }

  GET_METHOD( Integer, TaylorOrder )
    {
      return theTaylorOrder;
    }

  SET_METHOD( Integer, TaylorOrder )
    {
      theTaylorOrder = value;
    }

  ESSYNSStepper()
    :
    theESSYNSProcessPtr( NULLPTR ),
    theTaylorOrder( 1 )
    {
      ; 
    }
	    
  virtual ~ESSYNSStepper()
    {
      ;
    }

  virtual void initialize();
  virtual bool calculate();
    
  virtual GET_METHOD( Integer, Order )
    {
      return theTaylorOrder;
    }

  virtual InterpolantPtr createInterpolant( VariablePtr aVariable )
    {
      return new ESSYNSStepper::Interpolant( *this, aVariable );
    }

protected:

  Integer theSystemSize;
  Integer theTaylorOrder;
  ESSYNSProcessPtr   theESSYNSProcessPtr;
  std::vector<RealVector> theESSYNSMatrix;
  std::vector<VariableVector::size_type> theIndexVector;

  //  RealVector theK1;
};

#endif /* __ESSYNS_HPP */
