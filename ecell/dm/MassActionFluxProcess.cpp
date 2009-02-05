//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Tomoya Kitayama <tomo@e-cell.org>,
// E-Cell Project.
//

#include "libecs.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( MassActionFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( MassActionFluxProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, k );
    }

  MassActionFluxProcess()
    :
    k( 0.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, k );
  
  virtual void fire()
  {
    
    Real velocity( k * N_A );
    velocity *= getSuperSystem()->getSize();

    for( VariableReferenceVectorConstIterator 
	   s( theVariableReferenceVector.begin() );
	 s != theZeroVariableReferenceIterator; ++s )
      {
	VariableReference aVariableReference( *s );
	Integer aCoefficient( aVariableReference.getCoefficient() );
	do {
	  ++aCoefficient;
	  velocity *= aVariableReference.getMolarConc();
	} while( aCoefficient != 0 );
	
      }
    
    setFlux(velocity);
    
  }
  
  virtual void initialize()
  {
    Process::initialize();
    declareUnidirectional();
  }  

 protected:
  
  Real k;
    
};

LIBECS_DM_INIT( MassActionFluxProcess, Process );
