//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __INTERPOLANT_HPP
#define __INTERPOLANT_HPP

#include "libecs.hpp"

namespace libecs
{

  class Interpolant
  {
    friend class libecs::Stepper;


  public:

    class VariablePtrCompare
    {
    public:
      bool operator()( InterpolantCptr const aLhs, 
		       InterpolantCptr const aRhs ) const
      {
	return compare( aLhs->getVariable(), aRhs->getVariable() );
      }

      bool operator()( InterpolantCptr const aLhs,
		       VariableCptr const aRhs ) const
      {
	return compare( aLhs->getVariable(), aRhs );
      }

      bool operator()( VariableCptr const aLhs, 
		       InterpolantCptr const aRhs ) const
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


    Interpolant( VariablePtr const aVariable );

    virtual ~Interpolant();
    
    virtual const Real getDifference( RealParam aTime, RealParam anInterval )
    {
      return 0.0;
    }
    
    VariablePtr const getVariable() const
    {
      return theVariable;
    }

  private:

    VariablePtr const theVariable;
    
  };


  DECLARE_VECTOR( InterpolantPtr, InterpolantVector );

}



#endif /* __INTERPOLANT_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
