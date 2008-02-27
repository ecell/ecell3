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

#include "libecs/libecs.hpp"
#include "libecs/ContinuousProcess.hpp"
#include "libecs/Util.hpp"
#include "libecs/FullID.hpp"
#include "libecs/PropertyInterface.hpp"
#include "libecs/System.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/Variable.hpp"
#include "libecs/Interpolant.hpp"

LIBECS_DM_CLASS( QuasiDynamicFluxProcess, libecs::ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( QuasiDynamicFluxProcess, Process )
    {
      INHERIT_PROPERTIES( libecs::ContinuousProcess );
      PROPERTYSLOT_SET_GET( libecs::Integer, Irreversible );
      PROPERTYSLOT_SET_GET( libecs::Real, Vmax );
      PROPERTYSLOT_SET_GET( libecs::Polymorph, FluxDistributionList );
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

  SIMPLE_SET_GET_METHOD( libecs::Integer, Irreversible );
  SIMPLE_SET_GET_METHOD( libecs::Real, Vmax );

  SET_METHOD( libecs::Polymorph, FluxDistributionList )
    {
      const libecs::PolymorphVector aVector( value.asPolymorphVector() );
      
      theFluxDistributionVector.clear();
      for( libecs::PolymorphVectorConstIterator i( aVector.begin() );
	   i != aVector.end(); ++i )
	{
	  theFluxDistributionVector.push_back( ( *( findVariableReference( (*i).asString() ) ) ) );
	}      
    }
  
  GET_METHOD( libecs::Polymorph, FluxDistributionList )
    {
      libecs::PolymorphVector aVector;
      for( libecs::VariableReferenceVectorConstIterator
	     i( theFluxDistributionVector.begin() );
	   i != theFluxDistributionVector.end() ; ++i )
	{
	  libecs::FullID aFullID( (*i).getVariable()->getFullID() );
	  aVector.push_back( aFullID.getString() );
	}

      return aVector;
    }

  libecs::VariableReferenceVector getFluxDistributionVector()
    {
      return theFluxDistributionVector;
    }

  virtual void initialize()
    {
      libecs::Process::initialize();      
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

  libecs::VariableReferenceVector theFluxDistributionVector;
  libecs::Integer Irreversible;
  libecs::Real Vmax;

};


