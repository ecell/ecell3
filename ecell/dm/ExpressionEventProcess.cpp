//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2015 Keio University
//       Copyright (C) 2008-2015 RIKEN
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
// authors:
//    Yasuhiro Naito
//
// E-Cell Project.
//

#include "ExpressionProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_MIXIN( ExpressionEventProcess, Process,
                       ExpressionProcessBase )
{
public:

    LIBECS_DM_OBJECT( ExpressionEventProcess, Process )
    {
        INHERIT_PROPERTIES( _LIBECS_MIXIN_CLASS_ );
        INHERIT_PROPERTIES( Process );
        PROPERTYSLOT_SET_GET( String, Trigger );
        PROPERTYSLOT_SET_GET( String, Delay );
        PROPERTYSLOT_LOAD_SAVE( Polymorph, EventAssignmentList,
                                &ExpressionEventProcess::setEventAssignmentList,
                                &ExpressionEventProcess::getEventAssignmentList,
                                &ExpressionEventProcess::setEventAssignmentList,
                                &ExpressionEventProcess::getEventAssignmentList );
    }

    ExpressionEventProcess()
      : theFireTime( 0.0 ),
        theTriggerFlag( false ),
        theFireFlag( false )
    {
        //FIXME: additional properties:
        // Unidirectional     -> call declareUnidirectional() in initialize()
        //                                         if this is set
    }

    virtual ~ExpressionEventProcess()
    {
        // ; do nothing
    }

    SET_METHOD( libecs::String, Trigger )
    {
        theTrigger = value;
    }

    GET_METHOD( libecs::String, Trigger )
    {
        return theTrigger;
    }

    SET_METHOD( libecs::String, Delay )
    {
        theDelay = value;
    }

    GET_METHOD( libecs::String, Delay )
    {
        return theDelay;
    }

    SET_METHOD( libecs::Polymorph, EventAssignmentList )
    {
        if ( value.getType() != PolymorphValue::TUPLE )
        {
            THROW_EXCEPTION_INSIDE( ValueError,
                                    asString() + ": argument must be a tuple" );
        }
    
        typedef boost::range_const_iterator< PolymorphValue::Tuple >::type const_iterator;
        PolymorphValue::Tuple const& aTuple( value.as< PolymorphValue::Tuple const& >() );
    
        for ( const_iterator i( boost::begin( aTuple ) );
              i != boost::end( aTuple ); ++i )
        {
            if ( (*i).getType() != PolymorphValue::TUPLE )
            {
                THROW_EXCEPTION_INSIDE( ValueError,
                                        asString() + ": every element of the tuple "
                                        "must also be a tuple" );
            }
            
            PolymorphValue::Tuple const& anElem( (*i).as< PolymorphValue::Tuple const & >() );
            if ( anElem.size() != 2 )
            {
                THROW_EXCEPTION_INSIDE( ValueError,
                                        asString() + ": each element of the tuple "
                                        "must have exactly 2 elements" );
            }
            
            theEANameVector.push_back( anElem[ 0 ].as< String >() );
            theEventAssignmentMap[ anElem[ 0 ].as< String >() ] = 
                ExpressionProcessor( this, anElem[ 1 ].as< String >() );
        }
    }

    GET_METHOD( libecs::Polymorph, EventAssignmentList )
    {
        PolymorphVector aVector;
        aVector.reserve( theEventAssignmentMap.size() );
    
        for( EventAssignmentMap::const_iterator i(
                theEventAssignmentMap.begin() );
             i != theEventAssignmentMap.end() ; ++i )
        {
            aVector.push_back( boost::tuple< libecs::String, libecs::String >(
                (*i).first,
                (*i).second.getExpression() ) );
        }
    
        return Polymorph( aVector );
    }

    virtual void defaultSetProperty( libecs::String const& aPropertyName,
                                     libecs::Polymorph const& aValue )
    {
        return _LIBECS_MIXIN_CLASS_::defaultSetProperty( aPropertyName, aValue );
    }

    virtual libecs::Polymorph defaultGetProperty( libecs::String const& aPropertyName ) const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetProperty( aPropertyName );
    }

    virtual std::vector< libecs::String > defaultGetPropertyList() const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetPropertyList();
    }

    virtual libecs::PropertyAttributes
    defaultGetPropertyAttributes( libecs::String const& aPropertyName ) const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetPropertyAttributes( aPropertyName );
    }

    virtual void initialize()
    {
        theTriggerProcessor = ExpressionProcessor( this, theTrigger );
        theTriggerProcessor.initialize( getModel() );
        
        theDelayProcessor = ExpressionProcessor( this, theDelay );
        theDelayProcessor.initialize( getModel() );
        
        for( EANameVector::const_iterator aName( theEANameVector.begin() );
             aName != theEANameVector.end(); ++aName )
        {
            theEventAssignmentMap[ *aName ].initialize( getModel() );
        }

        Process::initialize();
        _LIBECS_MIXIN_CLASS_::initialize();
        
        for( VariableReferenceVector::const_iterator i(
                    getVariableReferenceVector().begin() );
             i != getVariableReferenceVector().end(); ++i )
        {
            if( i->getCoefficient() != 0 )
            {
                theVariableReference = *i;
                return;
            }
        }
        THROW_EXCEPTION_INSIDE(InitializationFailed, "No variable references with non-zero coefficients exist");
    }

    virtual void fire()
    { 
        if ( theTriggerFlag == false && theFireFlag == false )
        {
            if ( theTriggerProcessor.execute() != 0.0 )
            {
                theFireTime = getModel()->getCurrentTime() + 
                              theDelayProcessor.execute();
                theTriggerFlag = true;
            }
        }
        
        if ( theFireTime <= getModel()->getCurrentTime() && theTriggerFlag == true )
        {
            theFireFlag = true;
            theTriggerFlag = false;
            for( EANameVector::const_iterator aName = theEANameVector.begin();
                 aName != theEANameVector.end(); ++aName )
            {
                getVariableReference( *aName ).getVariable()->setValue( getVariableReference( *aName ).getCoefficient() * theEventAssignmentMap[ *aName ].execute() );
            }
        }
    }

protected:
    
    class ExpressionProcessor
    {
       public:
        
        ExpressionProcessor()
            : theCompiledCode( 0 ), theRecompileFlag( true )
        {}
        
        ExpressionProcessor( ExpressionEventProcess* thisProcess,
                             libecs::String anExpression )
            : theExpression( anExpression ), theCompiledCode( 0 ), theRecompileFlag( true )
        {
            setThisProcess( thisProcess );
        }
        
        void initialize( libecs::Model* const aModel )
        {
             setModel( aModel );
             compileExpression();
        }
        
        libecs::String getExpression() const
        {
            return theExpression;
        }
        
        void setThisProcess( ExpressionEventProcess* thisProcess )
        {
            _thisProcess = thisProcess;
        }

        void compileExpression()
        {
            try
            {
                _thisProcess->compileExpression( theExpression, theCompiledCode );
            }
            catch ( libecs::Exception const& e )
            {
                throw libecs::InitializationFailed( e, static_cast< Process * >( _thisProcess ) );
            }
        }

        void setModel( libecs::Model* const aModel )
        {
            theVirtualMachine.setModel( aModel );
        }
        
        libecs::Real execute() const
        {
            return theVirtualMachine.execute( *theCompiledCode );
        }
        
      private:
        libecs::String theExpression;
        const libecs::scripting::Code* theCompiledCode;
        mutable libecs::scripting::VirtualMachine theVirtualMachine;
        bool theRecompileFlag;
        ExpressionEventProcess* _thisProcess;
    };
    friend class ExpressionProcessor;

    typedef Loki::AssocVector< libecs::String, ExpressionProcessor,
                               std::less<const libecs::String> > EventAssignmentMap;
    typedef std::vector< libecs::String > EANameVector;

    libecs::String theTrigger;
    ExpressionProcessor theTriggerProcessor;

    libecs::String theDelay;
    ExpressionProcessor theDelayProcessor;

    libecs::Real theFireTime;
    bool theTriggerFlag;
    bool theFireFlag;

    EventAssignmentMap theEventAssignmentMap;
    EANameVector theEANameVector;

private:
    VariableReference theVariableReference;
    
};

LIBECS_DM_INIT( ExpressionEventProcess, Process );

