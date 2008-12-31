//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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

#include "QuasiDynamicFluxProcess.hpp"

USE_LIBECS;

LIBECS_DM_INIT( QuasiDynamicFluxProcess, Process );

LIBECS_DM_INIT_PROP_INTERFACE_DEF( QuasiDynamicFluxProcess )
{
    INHERIT_PROPERTIES( ContinuousProcess );
    PROPERTYSLOT_SET_GET( Integer, Irreversible );
    PROPERTYSLOT_SET_GET( Real, Vmax );
    PROPERTYSLOT_SET_GET( Polymorph, FluxDistributionList );
}

SET_METHOD_DEF( Integer, Irreversible, QuasiDynamicFluxProcess )
{
    irreversible_ = value;
}

GET_METHOD_DEF( Integer, Irreversible, QuasiDynamicFluxProcess )
{
    return irreversible_;
}

SET_METHOD_DEF( libecs::Real, Vmax, QuasiDynamicFluxProcess )
{
    vmax_ = value;
}

GET_METHOD_DEF( libecs::Real, Vmax, QuasiDynamicFluxProcess )
{
    return vmax_;
}

SET_METHOD_DEF( libecs::Polymorph, FluxDistributionList, QuasiDynamicFluxProcess )
{
    const libecs::PolymorphVector aVector( value.as< libecs::PolymorphVector >() );
    
    theFluxDistributionVector.clear();
    for( libecs::PolymorphVector::const_iterator i( aVector.begin() );
         i != aVector.end(); ++i )
    {
        theFluxDistributionVector.push_back( ( *( findVariableReference( (*i).as< libecs::String >() ) ) ) );
    }            
}

GET_METHOD_DEF( libecs::Polymorph, FluxDistributionList, QuasiDynamicFluxProcess )
{
    libecs::PolymorphVector aVector;
    for ( libecs::VariableReferenceVector::const_iterator i(
            theFluxDistributionVector.begin() );
          i != theFluxDistributionVector.end() ; ++i )
    {
        libecs::FullID aFullID( (*i).getVariable()->getFullID() );
        aVector.push_back( aFullID.asString() );
    }

    return aVector;
}

VariableReferenceVector QuasiDynamicFluxProcess::getFluxDistributionVector()
{
    return theFluxDistributionVector;
}

void QuasiDynamicFluxProcess::initialize()
{
    libecs::Process::initialize();            
    if( theFluxDistributionVector.empty() )
    {
        theFluxDistributionVector = theVariableReferenceVector;
    } 
}

void QuasiDynamicFluxProcess::fire()
{
    ; // do nothing
}
