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

#include <boost/format.hpp>

#include "Util.hpp"
#include "VariableReference.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "Exceptions.hpp"
#include "Variable.hpp"
#include "Model.hpp"

#include "Process.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( Process, Process );

  SET_METHOD_DEF( Polymorph, VariableReferenceList, Process )
  {
    const PolymorphVector aVector( value.asPolymorphVector() );
    for( PolymorphVectorConstIterator i( aVector.begin() );
	 i != aVector.end(); ++i )
      {
	const PolymorphVector anInnerVector( (*i).asPolymorphVector() );

	setVariableReference( anInnerVector );
      }

  }

  GET_METHOD_DEF( Polymorph, VariableReferenceList, Process )
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableReferenceVector.size() );
  
    for( VariableReferenceVectorConstIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end() ; ++i )
      {
	PolymorphVector anInnerVector;
	VariableReferenceCref aVariableReference( *i );

	// Tagname
	anInnerVector.push_back( aVariableReference.getName() );
	// FullID

	FullID aFullID( aVariableReference.getVariable()->getFullID() );
	aFullID.setEntityType( EntityType::NONE );
	anInnerVector.push_back( aFullID.getString() );
	// Coefficient
	anInnerVector.push_back( aVariableReference.getCoefficient() );
	// isAccessor
	anInnerVector.
	  push_back( static_cast<Integer>( aVariableReference.isAccessor() ) );

	aVector.push_back( anInnerVector );
      }

    return aVector;
  }

  SAVE_METHOD_DEF( Polymorph, VariableReferenceList, Process )
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableReferenceVector.size() );
  
    for( VariableReferenceVectorConstIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end() ; ++i )
      {
	PolymorphVector anInnerVector;
	VariableReferenceCref aVariableReference( *i );

	// (1) Variable reference name

	// convert back all variable reference ellipses to the default '_'.
	String aReferenceName( aVariableReference.getName() );

	if( VariableReference::
	    isEllipsisNameString( aReferenceName ) )
	  {
	    aReferenceName = VariableReference::DEFAULT_NAME;
	  }

	anInnerVector.push_back( aReferenceName );

	// (2) FullID

	FullID aFullID( aVariableReference.getVariable()->getFullID() );
	aFullID.setEntityType( EntityType::NONE );

	anInnerVector.push_back( aFullID.getString() );

	// (3) Coefficient and (4) IsAccessor
	const Integer aCoefficient( aVariableReference.getCoefficient() );
	const bool    anIsAccessorFlag( aVariableReference.isAccessor() );


	// include both if IsAccessor is non-default (not true).
	if( anIsAccessorFlag != true )
	  {
	    anInnerVector.push_back( aCoefficient );	    
	    anInnerVector.
	      push_back( static_cast<Integer>( anIsAccessorFlag ) );
	  }
	else
	  {
	    // output only the coefficient if IsAccessor has a 
	    // default value, and the coefficient is non-default.
	    if( aCoefficient != 0 )
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


  Process::Process() 
    :
    theZeroVariableReferenceIterator( theVariableReferenceVector.end() ),
    thePositiveVariableReferenceIterator( theVariableReferenceVector.end() ),
    theActivity( 0.0 ),
    thePriority( 0 ),
    theStepper( NULLPTR )
  {
    ; // do nothing
  }

  Process::~Process()
  {
    if( getStepper() != NULLPTR )
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
    if( theStepper != aStepper )
      {
	if( aStepper != NULLPTR )
	  {
	    aStepper->registerProcess( this );
	  }
	else
	  {
	    theStepper->removeProcess( this );
	  }

	theStepper = aStepper;
      }

  }

  VariableReference Process::getVariableReference( StringCref 
						   aVariableReferenceName )
  {
    VariableReferenceVectorConstIterator 
      anIterator( findVariableReference( aVariableReferenceName ) );

    if( anIterator != theVariableReferenceVector.end() )
      {
	return *anIterator;
      }
    else
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: VariableReference [" + aVariableReferenceName + 
			 "] not found in this Process." );
      }

  }

  void Process::removeVariableReference( StringCref aName )
  {
    theVariableReferenceVector.erase( findVariableReference( aName ) );
  }

  void Process::setVariableReference( PolymorphVectorCref aValue )
  {

    UnsignedInteger aVectorSize( aValue.size() );
    
    // Require at least a VariableReference name.
    if( aVectorSize == 0 )
      {
	THROW_EXCEPTION( ValueError, "Process [" + getFullID().getString()
			 + "]: ill-formed VariableReference given." );
      }

    const String aVariableReferenceName( aValue[0].asString() );

    // If it contains only the VariableReference name,
    // remove the VariableReference from this process
    if( aVectorSize == 1 )
      {
	removeVariableReference( aVariableReferenceName );
      }


    const String aFullIDString( aValue[1].asString() );
    const FullID aFullID( aValue[1].asString() );
    Integer      aCoefficient( 0 );
    
    // relative search; allow relative systempath
    SystemPtr aSystem( getSuperSystem()->
		       getSystem( aFullID.getSystemPath() ) );

    VariablePtr aVariable( aSystem->getVariable( aFullID.getID() ) );
    
    if( aVectorSize >= 3 )
      {
	aCoefficient = aValue[2].asInteger();
      }
    
    if( aVectorSize >= 4 )
      {
	const bool anIsAccessorFlag( static_cast<bool>
				     ( aValue[3].asInteger() ) );
	registerVariableReference( aVariableReferenceName, aVariable,
				   aCoefficient, anIsAccessorFlag );
      }
    else
      {
	registerVariableReference( aVariableReferenceName, aVariable, 
				   aCoefficient );
      }
    
  }


  void Process::registerVariableReference( StringCref aName, 
					   VariablePtr aVariable, 
					   IntegerParam aCoefficient,
					   const bool isAccessor )
  {
    String aVariableReferenceName( aName );

    if( VariableReference::isDefaultNameString( aVariableReferenceName ) )
      {
	try
	  {
	    Integer anEllipsisNumber( 0 );
	    if( ! theVariableReferenceVector.empty() )
	      {
		VariableReferenceVectorConstIterator 
		  aLastEllipsisIterator
		  ( std::max_element( theVariableReferenceVector.begin(), 
				      theVariableReferenceVector.end(), 
				      VariableReference::NameLess() ) );
		
		VariableReferenceCref aLastEllipsis( *aLastEllipsisIterator );
		
		anEllipsisNumber = aLastEllipsis.getEllipsisNumber();
		++anEllipsisNumber;
	      }
	    
	    aVariableReferenceName = VariableReference::ELLIPSIS_PREFIX + 
	      ( boost::format( "%03d" ) % anEllipsisNumber ).str();
	  }
	catch( const ValueError& )
	  {
	    ; // pass
	  }
      }

    if( findVariableReference( aVariableReferenceName ) != 
	theVariableReferenceVector.end() )
      {
	THROW_EXCEPTION( AlreadyExist,
			 "[" + getFullID().getString() + 
			 "]: VariableReference [" + aVariableReferenceName + 
			 "] already exists in this Process." );

      }

    VariableReference aVariableReference( aVariableReferenceName, 
					  aVariable, aCoefficient );
    theVariableReferenceVector.push_back( aVariableReference );


    //FIXME: can the following be moved to initialize()?
    updateVariableReferenceVector();
  }

  void Process::updateVariableReferenceVector()
  {
    // first sort by reference name
    std::sort( theVariableReferenceVector.begin(), 
	       theVariableReferenceVector.end(), 
	       VariableReference::Less() );

    // find the first VariableReference whose coefficient is 0,
    // and the first VariableReference whose coefficient is positive.
    std::pair
      <VariableReferenceVectorIterator, VariableReferenceVectorIterator> 
      aZeroRange( std::equal_range( theVariableReferenceVector.begin(), 
				    theVariableReferenceVector.end(), 
				    0, 
				    VariableReference::CoefficientLess()
				    ) );

    theZeroVariableReferenceIterator     = aZeroRange.first;
    thePositiveVariableReferenceIterator = aZeroRange.second;
  }



  VariableReferenceVectorIterator 
  Process::findVariableReference( StringCref aVariableReferenceName )
  {
    // well this is a linear search.. but this won't be used during simulation.
    for( VariableReferenceVectorIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end(); ++i )
      {
	if( (*i).getName() == aVariableReferenceName )
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
		   boost::bind2nd
		   ( boost::mem_fun_ref
		     ( &VariableReference::setIsAccessor ), false ) );
  }

  void Process::initialize()
  {
    ; // do nothing
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
