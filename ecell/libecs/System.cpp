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

#include <algorithm>

#include "Process.hpp"
#include "Model.hpp"
#include "Variable.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( System, System );

  /////////////////////// System


  // Property slots

  const Polymorph System::getSystemList() const
  {
    PolymorphVector aVector;
    aVector.reserve( getSystemMap().size() );

    for( SystemMapConstIterator i = getSystemMap().begin() ;
	 i != getSystemMap().end() ; ++i )
      {
	aVector.push_back( i->second->getID() );
      }

    return aVector;
  }

  const Polymorph System::getVariableList() const
  {
    PolymorphVector aVector;
    aVector.reserve( getVariableMap().size() );

    for( VariableMapConstIterator i( getVariableMap().begin() );
	 i != getVariableMap().end() ; ++i )
      {
	aVector.push_back( i->second->getID() );
      }

    return aVector;
  }

  const Polymorph System::getProcessList() const
  {
    PolymorphVector aVector;
    aVector.reserve( getProcessMap().size() );

    for( ProcessMapConstIterator i( getProcessMap().begin() );
	 i != getProcessMap().end() ; ++i )
      {
	aVector.push_back( i->second->getID() );
      }

    return aVector;
  }


  System::System()
    :
    theStepper( NULLPTR ),
    theSizeVariable( NULLPTR ),
    theModel( NULLPTR ),
    theEntityListChanged( false )
  {
    ; // do nothing
  }

  System::~System()
  {
    getStepper()->removeSystem( this );
    
    // delete Processes first.
    for( ProcessMapIterator i( theProcessMap.begin() );
	 i != theProcessMap.end() ; ++i )
      {
	delete i->second;
      }

    // then Variables.
    for( VariableMapIterator i( theVariableMap.begin() );
	 i != theVariableMap.end() ; ++i )
      {
	delete i->second;
      }

    // delete sub-systems.
    for( SystemMapIterator i( theSystemMap.begin() );
	 i != theSystemMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void System::setStepperID( StringCref anID )
  {
    theStepper = getModel()->getStepper( anID );
    theStepper->registerSystem( this );
  }

  const String System::getStepperID() const
  {
    return getStepper()->getID();
  }

  VariableCptr const System::findSizeVariable() const
  {
    try
      {
	return getVariable( "SIZE" );
      }
    catch( NotFoundCref )
      {
	SystemCptr const aSuperSystem( getSuperSystem() );

	// Prevent infinite looping.  But this shouldn't happen.
	if( aSuperSystem == this )
	  {
	    THROW_EXCEPTION( UnexpectedError, 
			     "While trying get a SIZE variable,"
			     " supersystem == this.  Probably a bug." );
	  }

	return aSuperSystem->findSizeVariable();
      }
  }

  GET_METHOD_DEF( Real, Size, System )
  {
    return theSizeVariable->getValue();
  }

  void System::configureSizeVariable()
  {
    theSizeVariable = findSizeVariable();
  }

  void System::initialize()
  {
    // do not need to call subsystems' initialize() -- the Model does this

    //
    // Variable::initialize()
    //
    for( VariableMapConstIterator i( getVariableMap().begin() );
	 i != getVariableMap().end() ; ++i )
      {
	i->second->initialize();
      }

    //
    // Process::initialize()
    //
    for( ProcessMapConstIterator i( getProcessMap().begin() );
	 i != getProcessMap().end() ; ++i )
      {
	ProcessPtr aProcessPtr( i->second );

	if( aProcessPtr->getStepper() == NULLPTR )
	  {
	    aProcessPtr->setStepper( getStepper() );
	  }

	aProcessPtr->initialize();
      }

    configureSizeVariable();
  }

  ProcessPtr System::getProcess( StringCref anID ) const
  {
    ProcessMapConstIterator i( getProcessMap().find( anID ) );

    if( i == getProcessMap().end() )
      {
	THROW_EXCEPTION( NotFound, 
			 "[" + getFullID().getString() + 
			 "]: Process [" + anID + 
			 "] not found in this System." );
      }

    return i->second;
  }


  VariablePtr System::getVariable( StringCref anID ) const
  {
    VariableMapConstIterator i( getVariableMap().find( anID ) );
    if( i == getVariableMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: Variable [" + anID + 
			 "] not found in this System.");
      }

    return i->second;
  }


  void System::registerSystem( SystemPtr aSystem )
  {
    const String anID( aSystem->getID() );

    if( getSystemMap().find( anID ) != getSystemMap().end() )
      {
	delete aSystem;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: System [" + anID + "] already exist." );
      }

    theSystemMap[ anID ] = aSystem;
    aSystem->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  SystemPtr System::getSystem( SystemPathCref aSystemPath ) const
  {
    if( aSystemPath.empty() )
      {
	return const_cast<SystemPtr>( this );
      }
    
    if( aSystemPath.isAbsolute() )
      {
	return getModel()->getSystem( aSystemPath );
      }

    SystemPtr const aNextSystem( getSystem( aSystemPath.front() ) );

    SystemPath aSystemPathCopy( aSystemPath );
    aSystemPathCopy.pop_front();

    return aNextSystem->getSystem( aSystemPathCopy );
  }
    

  SystemPtr System::getSystem( StringCref anID ) const
  {
    if( anID[0] == '.' )
      {
	const UnsignedInt anIDSize( anID.size() );

	if( anIDSize == 1 ) // == "."
	  {
	    return const_cast<SystemPtr>( this );
	  }
	else if( anID[1] == '.' && anIDSize == 2 ) // == ".."
	  {
	    if( isRootSystem() )
	      {
		THROW_EXCEPTION( NotFound,
				 "[" + getFullID().getString() + 
				 "]: cannot get a super system ('" + anID +
				 "') from a root system." );
	      }
	    
	    return getSuperSystem();
	  }
      }

    SystemMapConstIterator i( getSystemMap().find( anID ) );
    if( i == getSystemMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: System [" + anID + 
			 "] not found in this System." );
      }

    return i->second;
  }

  void System::notifyChangeOfEntityList()
  {
    //    getStepper()->getMasterStepper()->setEntityListChanged();
  }

  const SystemPath System::getSystemPath() const
  {
    if( isRootSystem() )
      {
	return SystemPath();
      }
    else
      {
	return Entity::getSystemPath();
      }
  }


  void System::registerProcess( ProcessPtr aProcess )
  {
    const String anID( aProcess->getID() );

    if( getProcessMap().find( anID ) != getProcessMap().end() )
      {
	delete aProcess;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Process [" + anID + "] already exist." );
      }

    theProcessMap[ anID ] = aProcess;
    aProcess->setSuperSystem( this );

    notifyChangeOfEntityList();
  }


  void System::registerVariable( VariablePtr aVariable )
  {
    const String anID( aVariable->getID() );

    if( getVariableMap().find( anID ) != getVariableMap().end() )
      {
	delete aVariable;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Variable [" + anID + "] already exist." );
      }

    theVariableMap[ anID ] = aVariable;
    aVariable->setSuperSystem( this );

    notifyChangeOfEntityList();
  }



} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
