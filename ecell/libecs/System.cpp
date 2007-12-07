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
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <algorithm>
#include <iostream>

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

  GET_METHOD_DEF( Polymorph, SystemList, System )
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

  GET_METHOD_DEF( Polymorph, VariableList, System )
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

  GET_METHOD_DEF( Polymorph, ProcessList, System )
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

  SET_METHOD_DEF( String, StepperID, System )
  {
    theStepper = getModel()->getStepper( value );
    theStepper->registerSystem( this );
  }

  GET_METHOD_DEF( String, StepperID, System )
  {
    return getStepper()->getID();
  }

  System::System()
    :
    theStepper( NULLPTR ),
    theSizeVariable( NULLPTR ),
    theModel( NULLPTR ),
    theEntityListChanged( false ),
    sizeInitialized( false )
  {
    ; // do nothing
  }

  System::~System()
  {
    if( getStepper() != NULLPTR )
      {
	getStepper()->removeSystem( this );
      }
    
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
    sizeInitialized = true;
  }

  void System::printSystems() const
  {
    String name = this->getFullID().getString();

    if (this->theSizeVariable)
      {
        std::cout << name << " with size " << this->theSizeVariable->getValue() << std::endl;
      }
    else
      {
        std::cout << "ERROR - " << name << " has no 'theSizeVariable'" << std::endl;
      }
    
    for (SystemMap::const_iterator i = theSystemMap.begin();
         i != theSystemMap.end();
         ++i)
      {
        i->second->printSystems();
      }

    return;
    }


  void System::initializeVariables()
  {
    //
    // Variable::initialize()
    //
    for( VariableMapConstIterator i( getVariableMap().begin() );
	 i != getVariableMap().end() ; ++i )
      {
	i->second->initialize();
      }
  }

  void System::checkProcessesHaveSteppers()
  {
    //
    // Set Process::theStepper.
    // Process::initialize() is called in Stepper::initialize()
    // 
    for( ProcessMapConstIterator i( getProcessMap().begin() );
	 i != getProcessMap().end() ; ++i )
      {
	ProcessPtr aProcessPtr( i->second );

	if( aProcessPtr->getStepper() == NULLPTR )
	  {
	    aProcessPtr->setStepper( getStepper() );
	  }
      }
  }

  void System::initialize()
  {

    // no need to call subsystems' initialize() -- the Model does this
    this->initializeVariables();
    this->checkProcessesHaveSteppers();

    configureSizeVariable();
    return;
  }

  void System::configureStepper()
  {
    if ( this->getStepper() == NULLPTR )
      {
        this->setStepperID( this->getSuperSystem()->getStepperID() );
      }
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
			 "]: System [" + anID + "] already exists." );
      }

    theSystemMap[ anID ] = aSystem;
    aSystem->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  void System::deleteSystem( SystemPtr aSystem )
  {

    // Assert the system is empty.
    uint systemSize =
      aSystem->getProcessMap().size() + 
      aSystem->getSystemMap().size() + 
      aSystem->getVariableMap().size();
    
    if (systemSize)
      {
        THROW_EXCEPTION( AssertionFailed,
                         "[" + getFullID().getString() + 
			 "]: System [" + aSystem->getID() + "] is not empty." );
      }


    const String anID( aSystem->getID() );

    // Find it and remove it...
    SystemMap::iterator i = theSystemMap.find( anID );

    if (i == theSystemMap.end() )
      {
        THROW_EXCEPTION( NotFound, 
                         "[" + getFullID().getString() +
                         "]: System [" + anID + "] does not exist.");
      }

    theSystemMap.erase(i);
    delete aSystem;

    getModel()->setDirtyBit();
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
    const String::size_type anIDSize( anID.size() );

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
			 "]: Process [" + anID + "] already exists." );
      }

    theProcessMap[ anID ] = aProcess;
    aProcess->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  void System::deleteProcess( ProcessPtr aProcess)
  {
    // Remove it from the System.
    const String anID( aProcess->getID() );
    ProcessMap::iterator i = theProcessMap.find( anID );
    
    if (i == theProcessMap.end() )
      {
        THROW_EXCEPTION( NotFound, 
                         "[" + getFullID().getString() +
                         "]: Process [" + anID + "] does not exist.");
      }
    
    delete i->second;
    theProcessMap.erase(i);

    getModel()->setDirtyBit();
  }

  void System::registerVariable( VariablePtr aVariable )
  {
    const String anID( aVariable->getID() );

    if( getVariableMap().find( anID ) != getVariableMap().end() )
      {
	delete aVariable;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Variable [" + anID + "] already exists." );
      }

    theVariableMap[ anID ] = aVariable;
    aVariable->setSuperSystem( this );

    notifyChangeOfEntityList();
  }


  void System::deleteVariable( VariablePtr aVariable)
  {
    const String anID( aVariable->getID() );
    VariableMap::iterator i = theVariableMap.find( anID );

    if (i == theVariableMap.end() )
      {
        THROW_EXCEPTION( NotFound, 
                         "[" + getFullID().getString() +
                         "]: Variable [" + anID + "] does not exist.");
      }
    
    delete i->second;
    theVariableMap.erase( i );

    getModel()->setDirtyBit();
  }

  void System::removeContents()
  {
    ModelPtr theModel = getModel();

    for( SystemMapIterator i = theSystemMap.begin();
         i != theSystemMap.end();
         ++i)
      {
        SystemPtr aSystem = i->second;
        FullID aFullID = aSystem->getFullID();
        String theFullIDString = aFullID.getString();
        theModel->removeEntity( aFullID );
      }
    
    for (ProcessMapIterator i = theProcessMap.begin();
         i != theProcessMap.end();
         ++i)
      {
        ProcessPtr aProcessPtr = i->second;
        FullID aFullID = aProcessPtr->getFullID();
        String theFullString = aFullID.getString();
        theModel->removeEntity( aFullID );
      }
    
    for (VariableMapIterator i = theVariableMap.begin();
         i != theVariableMap.end();
         ++i)
      {
        VariablePtr aVariablePtr = i->second;
        FullID aFullID = aVariablePtr->getFullID();
        String theFullIDString = aFullID.getString();
        theModel->removeEntity( aFullID );
      }
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
