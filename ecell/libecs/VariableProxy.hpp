//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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

#ifndef __VARIABLEPROXY_HPP
#define __VARIABLEPROXY_HPP

#include "libecs.hpp"

namespace libecs
{

  class VariableProxy
  {
    friend class libecs::Stepper;


  public:

    class VariablePtrCompare
    {
    public:
      bool operator()( VariableProxyCptr const aLhs, 
		       VariableProxyCptr const aRhs ) const
      {
	return compare( aLhs->getVariable(), aRhs->getVariable() );
      }

      bool operator()( VariableProxyCptr const aLhs,
		       VariableCptr const aRhs ) const
      {
	return compare( aLhs->getVariable(), aRhs );
      }

      bool operator()( VariableCptr const aLhs, 
		       VariableProxyCptr const aRhs ) const
      {
	return compare( aLhs, aRhs->getVariable() );
      }

    private:

      // if statement can be faster than returning an expression directly
      inline static bool compare( VariableCptr const aLhs, 
				  VariableCptr const aRhs )
      {
	if( aLhs < aRhs )
	  {
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }


    };


    VariableProxy( VariablePtr const aVariable );

    virtual ~VariableProxy();
    
    virtual const Real getDifference( RealCref aTime, RealCref anInterval )
    {
      return 0.0;
    }
    
    VariablePtr const getVariable() const
    {
      return theVariable;
    }

    virtual const StepperPtr getStepperPtr()
    {
      return NULLPTR;
    }

  private:

    VariablePtr const theVariable;
    
  };


  DECLARE_VECTOR( VariableProxyPtr, VariableProxyVector );

}



#endif /* __VARIABLEPROXY_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
