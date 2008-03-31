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

namespace libecs
{

LIBECS_DM_INIT_STATIC( Process, Process );

LOAD_METHOD_DEF( VariableReferenceList, Process )
{
    const PolymorphVector v( value.asPolymorphVector() );
    for ( PolymorphVector::const_iterator i( v.begin() ); i != v.end(); ++i )
    {
        const PolymorphVector aValue( ( *i ).asPolymorphVector() );
        size_t aVectorSize( aValue.size() );

        // Require at least a VariableReference name.
        if ( aVectorSize == 0 )
        {
            THROW_EXCEPTION( ValueError, "wrong VariableReference given." );
        }

        const String varRefName( aValue[0].asString() );

        // If it contains only the VariableReference name,
        // remove the VariableReference from this process
        if ( aVectorSize == 1 )
        {
            removeVariableReference( varRefName );
        }


        const FullID aFullID( FullID::parse( aValue[1].asString() ) );
        Integer      aCoefficient( 0 );

        // relative search; allow relative systempath
        System* aSystem( getEnclosingSystem()->getSystem(
                aFullID.getSystemPath() ) );

        Variable* aVariable( aSystem->getVariable( aFullID.getID() ) );

        if ( aVectorSize >= 3 )
        {
            aCoefficient = aValue[2].asInteger();
        }

        if ( aVectorSize >= 4 )
        {
            const bool anIsAccessorFlag( aValue[3].asInteger() != 0 );
            registerVariableReference( varRefName, aVariable,
                                       aCoefficient, anIsAccessorFlag );
        }
        else
        {
            registerVariableReference( varRefName, aVariable,
                                       aCoefficient );
        }
    }

}

SAVE_METHOD_DEF( VariableReferenceList, Process )
{
    PolymorphVector aVector;
    aVector.reserve( varRefs_.size() );

    for ( VarRefs::const_iterator i( varRefs_.begin() );
            i != varRefs_.end() ; ++i )
    {
        PolymorphVector anInnerVector;
        const VariableReference& varRef( *i );

        // (1) Variable reference name

        // convert back all variable reference ellipses to the default '_'.
        String aReferenceName( varRef.getName() );

        if ( VariableReference::
                isEllipsisNameString( aReferenceName ) )
        {
            aReferenceName = VariableReference::DEFAULT_NAME;
        }

        anInnerVector.push_back( aReferenceName );

        // (2) FullID

        FullID aFullID( varRef.getVariable()->getFullID() );

        anInnerVector.push_back( aFullID.asString() );

        // (3) Coefficient and (4) IsAccessor
        const Integer aCoefficient( varRef.getCoefficient() );
        const bool    anIsAccessorFlag( varRef.isAccessor() );


        // include both if IsAccessor is non-default (not true).
        if ( anIsAccessorFlag != true )
        {
            anInnerVector.push_back( aCoefficient );
            anInnerVector.
            push_back( static_cast<Integer>( anIsAccessorFlag ) );
        }
        else
        {
            // output only the coefficient if IsAccessor has a
            // default value, and the coefficient is non-default.
            if ( aCoefficient != 0 )
            {
                anInnerVector.push_back( aCoefficient );
            }
            else
            {
                ; // do nothing -- both are the default
            }
        }

        aVector.push_back( anInnerVector );
    }

    return aVector;
}


void Process::startup()
{
    stepper_ = NULLPTR;
    activity_ = 0.0;
    priority_ = 0;
}

Process::~Process()
{
    if ( getStepper() != NULLPTR )
    {
        getStepper()->removeProcess( this );
    }
}


SET_METHOD_DEF( String, StepperID, Process )
{
    StepperPtr aStepperPtr( getModel()->getStepper( value ) );

    setStepper( aStepperPtr );
}

GET_METHOD_DEF( String, StepperID, Process )
{
    return getStepper()->getID();
}


void Process::setStepper( StepperPtr const aStepper )
{
    if ( stepper_ != aStepper )
    {
        if ( aStepper != NULLPTR )
        {
            aStepper->registerProcess( this );
        }
        else
        {
            stepper_->removeProcess( this );
        }

        stepper_ = aStepper;
    }

}

const VariableReference& Process::getVariableReference( const String&
        varRefName ) const
{
    VarRefs::const_iterator
    anIterator( findVariableReference( varRefName ) );

    if ( anIterator != varRefs_.end() )
    {
        return *anIterator;
    }
    else
    {
        THROW_EXCEPTION( NotFound,
                         "VariableReference [" + varRefName +
                         "] not found." );
    }

}

void Process::removeVariableReference( const String& aName )
{
    varRefs_.erase( findVariableReference( aName ) );
}

void Process::registerVariableReference( const String& name,
        Variable* var,
        IntegerParam coef,
        const bool isAccessor )
{
    String varRefName( name );

    if ( VariableReference::isDefaultNameString( varRefName ) )
    {
        try
        {
            Integer anEllipsisNumber( 0 );
            if ( ! varRefs_.empty() )
            {
                VarRefs::const_iterator aLastEllipsisIterator(
                        std::max_element(
                            varRefs_.begin(), varRefs_.end(),
                            VariableReference::NameLess() ) );
                const VariableReference& aLastEllipsis( *aLastEllipsisIterator );
                anEllipsisNumber = aLastEllipsis.getEllipsisNumber();
                ++anEllipsisNumber;
            }

            varRefName = VariableReference::ELLIPSIS_PREFIX +
                ( boost::format( "%03d" ) % anEllipsisNumber ).str();
        }
        catch ( const ValueError& )
        {
            ; // pass
        }
    }

    if ( findVariableReference( varRefName ) != varRefs_.end() )
    {
        THROW_EXCEPTION( AlreadyExist,
                         "VariableReference [" + varRefName +
                         "] already exists." );

    }

    varRefs_.push_back(
            coef < 0 ? 0: ( coef > 0 ? 2: 1 ),
            VariableReference( varRefName, var, coef, isAccessor ) );
}

Process::VarRefs::iterator
Process::findVariableReference( const String& varRefName )
{
    // well this is a linear search.. but this won't be used during simulation.
    for ( VarRefs::iterator i( varRefs_.begin() );
            i != varRefs_.end(); ++i )
    {
        if ( ( *i ).getName() == varRefName )
        {
            return i;
        }
    }

    return varRefs_.end();
}

Process::VarRefs::const_iterator
Process::findVariableReference( const String& varRefName ) const
{
    // well this is a linear search.. but this won't be used during simulation.
    for ( VarRefs::const_iterator i( varRefs_.begin() );
            i != varRefs_.end(); ++i )
    {
        if ( ( *i ).getName() == varRefName )
        {
            return i;
        }
    }

    return varRefs_.end();
}

void Process::declareUnidirectional()
{
    VarRefVectorRange positiveRefs( getPositiveVariableReferences() );

    std::for_each( positiveRefs.begin(), positiveRefs.end(),
           boost::bind2nd( boost::mem_fun_ref(
                    &VariableReference::setIsAccessor ), false ) );
}

const bool Process::isDependentOn( const Process* proc ) const
{
    VarRefVectorCRange aVarRefs( proc->getVariableReferences() );

    for ( VarRefs::const_iterator i( varRefs_.begin() );
            i != varRefs_.end() ; ++i )
    {
        const VariableReference& varRef1( *i );

        for ( VarRefs::const_iterator j( aVarRefs.begin() );
                j != aVarRefs.end(); ++j )
        {
            const VariableReference& varRef2( *j );

            if ( varRef1.getVariable() == varRef2.getVariable() &&
                    varRef1.isAccessor() && varRef2.isMutator() )

            {
                return true;
            }
        }
    }

    return false;
}

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
