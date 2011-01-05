//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "Util.hpp"
#include "VariableReference.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "Exceptions.hpp"
#include "Variable.hpp"
#include "Model.hpp"

#include "Process.hpp"

#include <boost/format.hpp>
#include <boost/bind.hpp>

namespace libecs
{

LIBECS_DM_INIT_STATIC( Process, Process );

SET_METHOD_DEF( Polymorph, VariableReferenceList, Process )
{
    if ( value.getType() != PolymorphValue::TUPLE )
    {
        THROW_EXCEPTION_INSIDE( ValueError,
                                asString() + ": argument must be a tuple" );
    }

    typedef boost::range_const_iterator< PolymorphValue::Tuple >::type const_iterator;
    PolymorphValue::Tuple const& aTuple( value.as< PolymorphValue::Tuple const& >() );

    for ( const_iterator i( boost::begin( aTuple ) ); i != boost::end( aTuple );
          ++i )
    {
        if ( (*i).getType() != PolymorphValue::TUPLE )
        {
            THROW_EXCEPTION_INSIDE( ValueError,
                                    asString() + ": every element of the tuple "
                                    "must also be a tuple" );
        }
        PolymorphValue::Tuple const& anElem( (*i).as< PolymorphValue::Tuple const & >() );
        if ( anElem.size() < 2 )
        {
            THROW_EXCEPTION_INSIDE( ValueError,
                                    asString() + ": each element of the tuple "
                                    "must have at least 4 elements" );
        }
        registerVariableReference(
            anElem[ 0 ].as< String >(),
            FullID( anElem[ 1 ].as< String >() ),
            anElem.size() > 2 ? anElem[ 2 ].as< Integer >(): 0l,
            ( anElem.size() > 3 ?
                anElem[ 3 ].as< Integer >(): 1l ) != 0 ? true: false );
    }
}

GET_METHOD_DEF( Polymorph, VariableReferenceList, Process )
{
    PolymorphVector aVector;
    aVector.reserve( theVariableReferenceVector.size() );

    for( VariableReferenceVector::const_iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end() ; ++i )
    {
        VariableReference const& aVariableReference( *i );
        FullID aFullID( aVariableReference.getVariable()->getFullID() );
        aFullID.setEntityType( EntityType::NONE );

        aVector.push_back( boost::tuple< String, String, Integer, Integer >(
            aVariableReference.getName(),
            aFullID.asString(),
            aVariableReference.getCoefficient(),
            aVariableReference.isAccessor() ) );
    }

    return Polymorph( aVector );
}

SAVE_METHOD_DEF( Polymorph, VariableReferenceList, Process )
{
    PolymorphVector aVector;
    aVector.reserve( theVariableReferenceVector.size() );

    for( VariableReferenceVector::const_iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end() ; ++i )
    {
        VariableReference const& aVariableReference( *i );

        // (1) Variable reference name

        // convert back all variable reference ellipses to the default '_'.
        String aReferenceName( aVariableReference.getName() );

        if( VariableReference::isEllipsisNameString( aReferenceName ) )
        {
            aReferenceName = VariableReference::DEFAULT_NAME;
        }

        // (2) FullID

        FullID aFullID( aVariableReference.getVariable()->getFullID() );
        aFullID.setEntityType( EntityType::NONE );


        // (3) Coefficient and (4) IsAccessor
        const Integer aCoefficient( aVariableReference.getCoefficient() );
        const bool        anIsAccessorFlag( aVariableReference.isAccessor() );

        // include both if IsAccessor is non-default (not true).
        if( anIsAccessorFlag != true )
        {
            aVector.push_back( boost::tuple< String, String, Integer, Integer >(
                aReferenceName,
                aFullID.asString(),
                aCoefficient,
                static_cast<Integer>( anIsAccessorFlag ) ) );
        }
        else
        {
            // output only the coefficient if IsAccessor has a 
            // default value, and the coefficient is non-default.
            if( aCoefficient != 0 )
            {
                aVector.push_back( boost::tuple< String, String, Integer >(
                    aReferenceName,
                    aFullID.asString(),
                    aCoefficient ) );
            }
            else
            {
                aVector.push_back( boost::tuple< String, String >(
                    aReferenceName,
                    aFullID.asString() ) );
            }
        }
    }

    return Polymorph( aVector );
}


Process::Process() 
    : theZeroVariableReferenceIterator( theVariableReferenceVector.end() ),
      thePositiveVariableReferenceIterator( theVariableReferenceVector.end() ),
      theActivity( 0.0 ),
      thePriority( 0 ),
      theStepper( 0 ),
      theNextSerial( 1 )
{
    ; // do nothing
}


Process::~Process()
{
}


SET_METHOD_DEF( String, StepperID, Process )
{
    Stepper* aStepperPtr( getModel()->getStepper( value ) );

    setStepper( aStepperPtr );
}

GET_METHOD_DEF( String, StepperID, Process )
{
    Stepper const* aStepper( getStepper() );
    return aStepper ? aStepper->getID(): "";
}


void Process::setStepper( Stepper* aStepper )
{
    if( theStepper != aStepper )
    {
        if( aStepper )
        {
            aStepper->registerProcess( this );
        }
        else
        {
            theStepper->unregisterProcess( this );
        }

        theStepper = aStepper;
    }
}

VariableReference const&
Process::getVariableReference( Integer anID ) const
{
    VariableReferenceVector::const_iterator anIterator(
            findVariableReference( anID ) );

    if( anIterator != theVariableReferenceVector.end() )
    {
        return *anIterator;
    }
    else
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": VariableReference #"
                                + stringCast( anID )
                                + " not found in this Process" );
    }
}

VariableReference const&
Process::getVariableReference( String const& aVariableReferenceName ) const
{
    VariableReferenceVector::const_iterator anIterator(
            findVariableReference( aVariableReferenceName ) );

    if( anIterator != theVariableReferenceVector.end() )
    {
        return *anIterator;
    }
    else
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": VariableReference ["
                                + aVariableReferenceName
                                + "] not found in this Process" );
    }
}

bool Process::removeVariableReference( Integer anID, bool raiseException )
{
    VariableReferenceVector::iterator i( findVariableReference( anID ) );
    if ( i == theVariableReferenceVector.end() )
    {
        if ( !raiseException )
        {
            return false;
        }
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": VariableReference #"
                                + stringCast( anID )
                                + " not found in this Process" );
    }
    theVariableReferenceVector.erase( i );
    return true;
}

bool Process::removeVariableReference( String const& aName, bool raiseException  )
{
    bool aIsRemoved( false );

    for ( VariableReferenceVector::size_type
            i( 0 ), e ( theVariableReferenceVector.size() );
            i < e; )
    {
        if ( theVariableReferenceVector[ i ].getName() == aName )
        {
            theVariableReferenceVector.erase( theVariableReferenceVector.begin() + i );
            aIsRemoved = true;
        }
        else
        {
            ++i;
        }
    }

    if ( !aIsRemoved && raiseException )
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": VariableReference ["
                                + aName
                                + "] not found in this Process" );
    }

    return aIsRemoved;
}

bool Process::removeVariableReference( Variable const* aVariable, bool raiseException )
{
    bool aIsRemoved( false );

    for ( VariableReferenceVector::size_type
            i( 0 ), e( theVariableReferenceVector.size() );
            i < e; )
    {
        if ( theVariableReferenceVector[ i ].getVariable() == aVariable )
        {
            theVariableReferenceVector.erase( theVariableReferenceVector.begin() + i );
            aIsRemoved = true;
            --e;
        }
        else
        {
            ++i;
        }
    }

    if ( !aIsRemoved && raiseException )
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": VariableReference ["
                                + aVariable->asString()
                                + "] not found in this Process" );
    }

    return aIsRemoved;
}

Integer Process::registerVariableReference( FullID const& aFullID,
                                            Integer aCoefficient,
                                            const bool isAccessor )
{
    theVariableReferenceVector.push_back(
            VariableReference(
                theNextSerial, aFullID, aCoefficient, isAccessor ) );
    return theNextSerial++;
}


Integer Process::registerVariableReference( String const& aName,
                                            FullID const& aFullID,
                                            Integer aCoefficient,
                                            const bool isAccessor )
{
    theVariableReferenceVector.push_back(
            VariableReference(
                theNextSerial, aFullID, aCoefficient, isAccessor ) );
    theVariableReferenceVector.back().setName( aName );
    return theNextSerial++;
}

Integer Process::registerVariableReference( Variable* aVariable,
                                            Integer aCoefficient,
                                            const bool isAccessor )
{
    theVariableReferenceVector.push_back(
            VariableReference(
                theNextSerial, aVariable, aCoefficient, isAccessor ) );
    return theNextSerial++;
}


Integer Process::registerVariableReference( String const& aName,
                                            Variable* aVariable,
                                            Integer aCoefficient,
                                            const bool isAccessor )
{
    theVariableReferenceVector.push_back(
            VariableReference(
                theNextSerial, aVariable, aCoefficient, isAccessor ) );
    theVariableReferenceVector.back().setName( aName );
    return theNextSerial++;
}


void Process::resolveVariableReferences()
{
    if ( !getSuperSystem() )
    {
        THROW_EXCEPTION_INSIDE( IllegalOperation,
                                asString() + ": process is not associated to "
                                "any system" );
    }

    Integer anEllipsisNumber( 0 );

    for ( VariableReferenceVector::iterator
            i( theVariableReferenceVector.begin() ),
            e( theVariableReferenceVector.end() );
          i != e; ++i )
    {
        VariableReference& aVarRef( *i );

        if( aVarRef.isDefaultName() )
        {
            aVarRef.setName( VariableReference::ELLIPSIS_PREFIX + 
                ( boost::format( "%03d" ) % anEllipsisNumber ).str() );
            anEllipsisNumber++;
        }

        if ( !aVarRef.getVariable() )
        {
            // relative search; allow relative systempath
            FullID const& aFullID( aVarRef.getFullID() );
            if ( aFullID.isValid() )
            {
                System* aSystem( 0 );
                try
                {
                    aSystem = getSuperSystem()->getSystem( aFullID.getSystemPath() );
                }
                catch ( BadSystemPath const& )
                {
                }

                if ( !aSystem )
                {
                    THROW_EXCEPTION_INSIDE( IllegalOperation,
                                            asString() + ": system path ["
                                            + aFullID.getSystemPath().asString()
                                            + "] could not be resolved" );
                }

                aVarRef.setVariable( aSystem->getVariable( aFullID.getID() ) );
            }
            else
            {
                THROW_EXCEPTION_INSIDE( IllegalOperation,
                                        asString() + ": variable reference #"
                                        + boost::lexical_cast<std::string>( aVarRef.getSerial() )
                                        + " could not be resolved" );
            }
        }
        else
        {
            aVarRef.setFullID( aVarRef.getVariable()->getFullID() );
        }
    }
}

void Process::updateVariableReferenceVector()
{
    // first sort by reference name
    std::sort( theVariableReferenceVector.begin(), 
               theVariableReferenceVector.end(), 
               VariableReference::Less() );

    // find the first VariableReference whose coefficient is 0,
    // and the first VariableReference whose coefficient is positive.
    std::pair< VariableReferenceVector::iterator,
               VariableReferenceVector::iterator > aZeroRange(
        std::equal_range( theVariableReferenceVector.begin(), 
                          theVariableReferenceVector.end(), 
                          0, VariableReference::CoefficientLess() ) );

    theZeroVariableReferenceIterator     = aZeroRange.first;
    thePositiveVariableReferenceIterator = aZeroRange.second;
}



Process::VariableReferenceVector::iterator
Process::findVariableReference( String const& aVariableReferenceName )
{
    // well this is a linear search.. but this won't be used during simulation.
    for( VariableReferenceVector::iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end(); ++i )
    {
        if( (*i).getName() == aVariableReferenceName )
        {
            return i;
        }
    }

    return theVariableReferenceVector.end();
}


Process::VariableReferenceVector::const_iterator 
Process::findVariableReference( String const& aVariableReferenceName ) const
{
    // well this is a linear search.. but this won't be used during simulation.
    for( VariableReferenceVector::const_iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end(); ++i )
    {
        if( (*i).getName() == aVariableReferenceName )
        {
            return i;
        }
    }

    return theVariableReferenceVector.end();
}

Process::VariableReferenceVector::iterator
Process::findVariableReference( Integer anID )
{
    // well this is a linear search.. but this won't be used during simulation.
    for( VariableReferenceVector::iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end(); ++i )
    {
        if( (*i).getSerial() == anID )
        {
            return i;
        }
    }

    return theVariableReferenceVector.end();
}


Process::VariableReferenceVector::const_iterator 
Process::findVariableReference( Integer anID ) const
{
    // well this is a linear search.. but this won't be used during simulation.
    for( VariableReferenceVector::const_iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end(); ++i )
    {
        if( (*i).getSerial() == anID )
        {
            return i;
        }
    }

    return theVariableReferenceVector.end();
}



void Process::declareUnidirectional()
{
    std::for_each( thePositiveVariableReferenceIterator,
                   theVariableReferenceVector.end(),
                   boost::bind2nd( boost::mem_fun_ref( &VariableReference::setIsAccessor ), false ) );
}



bool Process::isDependentOn( Process const* aProcessPtr ) const
{
    VariableReferenceVector const& aVariableReferenceVector(
            aProcessPtr->getVariableReferenceVector() );
    
    for( VariableReferenceVector::const_iterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end() ; ++i )
    {
        VariableReference const& aVariableReference1( *i );

        for( VariableReferenceVector::const_iterator j(
                aVariableReferenceVector.begin() );
             j != aVariableReferenceVector.end(); ++j )
        {
            VariableReference const& aVariableReference2( *j );
            
            if( aVariableReference1.getVariable() == 
                    aVariableReference2.getVariable() && 
                aVariableReference1.isAccessor() && 
                aVariableReference2.isMutator() )
            {
                return true;
            }
        }
    }

    return false;
}

void Process::preinitialize()
{
    resolveVariableReferences();
    updateVariableReferenceVector(); 
} 

void Process::initialize()
{
    if( !getStepper() )
    {
        THROW_EXCEPTION_INSIDE( InitializationFailed,
                                "No stepper is connected with [" +
                                asString() + "]." );
    }
}

void Process::addValue( VariableReference const& aVarRef, Real value )
{
    aVarRef.getVariable()->addValue( aVarRef.getCoefficient() * value );
}

void Process::addValue( libecs::Param<Real>::type aValue )
{
    setActivity( aValue );

    // Increase or decrease variables, skipping zero coefficients.
    std::for_each( theVariableReferenceVector.begin(),
                   theZeroVariableReferenceIterator,
                   boost::bind( &VariableReference::addValue, _1, aValue ) );

    std::for_each( thePositiveVariableReferenceIterator,
                   theVariableReferenceVector.end(),
                   boost::bind( &VariableReference::addValue, _1, aValue ) );
}

void Process::detach()
{
    if ( theStepper )
    {
        try { theStepper->unregisterProcess( this ); } catch ( NotFound const& ) {}
    }
    Entity::detach();
}

} // namespace libecs
