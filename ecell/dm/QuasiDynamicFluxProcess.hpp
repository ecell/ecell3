//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
#include "libecs.hpp"
#include "ContinuousProcess.hpp"
#include "Util.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( QuasiDynamicFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( QuasiDynamicFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );
      PROPERTYSLOT_SET_GET( Integer, Irreversible );
      PROPERTYSLOT_SET_GET( Real, Vmax );
      PROPERTYSLOT_SET_GET( Polymorph, FluxDistributionList );
    }

  QuasiDynamicFluxProcess()
    :    
    Irreversible( 0 ),
    Vmax( 0 )
    {
      theFluxDistributionVector.reserve( 0 );
    }

  ~QuasiDynamicFluxProcess()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Integer, Irreversible );
  SIMPLE_SET_GET_METHOD( Real, Vmax );

  SET_METHOD( Polymorph, FluxDistributionList )
    {
      const PolymorphVector aVector( value.asPolymorphVector() );
      
      theFluxDistributionVector.clear();
      for( PolymorphVectorConstIterator i( aVector.begin() );
	   i != aVector.end(); ++i )
	{
	  theFluxDistributionVector.push_back( ( *( findVariableReference( (*i).asString() ) ) ) );
	}      
    }
  
  GET_METHOD( Polymorph, FluxDistributionList )
    {
      PolymorphVector aVector;
      for( VariableReferenceVectorConstIterator
	     i( theFluxDistributionVector.begin() );
	   i != theFluxDistributionVector.end() ; ++i )
	{
	  FullID aFullID( (*i).getVariable()->getFullID() );
	  aVector.push_back( aFullID.getString() );
	}

      return aVector;
    }

  VariableReferenceVector getFluxDistributionVector()
    {
      return theFluxDistributionVector;
    }

  virtual void initialize()
    {
      Process::initialize();      
      if( theFluxDistributionVector.empty() )
	{
	  theFluxDistributionVector = theVariableReferenceVector;
	} 
    }

  virtual void fire()
    {
      ; // do nothing
    }
  
 protected:

  VariableReferenceVector theFluxDistributionVector;
  Integer Irreversible;
  Real Vmax;

};


