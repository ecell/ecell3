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

#include "Util.hpp"
#include "EntityType.hpp"
#include "StepperMaker.hpp"
#include "VariableMaker.hpp"
#include "ProcessMaker.hpp"
#include "SystemMaker.hpp"
#include "LoggerBroker.hpp"
#include "Stepper.hpp"
#include "SystemStepper.hpp"

#include "Model.hpp"


namespace libecs
{

  ////////////////////////// Model

  Model::Model()
    :
    theCurrentTime( 0.0 ),
    theLoggerBroker(),
    theRootSystemPtr(0),
    theSystemStepper(),
    theStepperMaker(),
    theSystemMaker(),
    theVariableMaker(),
    theProcessMaker(),
    theRunningFlag( false ),
    theDirtyBit( false )
  {
    theLoggerBroker.setModel( this );
    // initialize theRootSystem
    theRootSystemPtr = getSystemMaker().make( "System" );
    theRootSystemPtr->setModel( this );
    theRootSystemPtr->setID( "/" );
    theRootSystemPtr->setName( "The Root System" );
    // super system of the root system is itself.
    theRootSystemPtr->setSuperSystem( theRootSystemPtr );
   
    // initialize theSystemStepper
    theSystemStepper.setModel( this );
    theSystemStepper.setID( "___SYSTEM" );
    theScheduler.addEvent(
      StepperEvent( getCurrentTime()
                    + theSystemStepper.getStepInterval(),
                    &theSystemStepper ) );

    theLastStepper = &theSystemStepper;
  }

  Model::~Model()
  {
    delete theRootSystemPtr;
  }


  void Model::flushLoggers()
  {
    theLoggerBroker.flush();
  }


  PolymorphMap Model::getClassInfo( StringCref aClassType, StringCref aClassname, Integer forceReload )
  {
	const void* (*InfoPtrFunc)();
     
    if ( aClassType == "Stepper" )
      {
        InfoPtrFunc = getStepperMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
      }
    else
      {
        EntityType anEntityType( aClassType );
        if ( anEntityType.getType() == EntityType::VARIABLE )
	      {    
		    InfoPtrFunc = getVariableMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
	      }
        else if ( anEntityType.getType() == EntityType::PROCESS )
	      {
		    InfoPtrFunc = getProcessMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
	      }
        else if ( anEntityType.getType() == EntityType::SYSTEM )
	      {
		    InfoPtrFunc = getSystemMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
	      }
        else 
	      {
		    THROW_EXCEPTION( InvalidEntityType,
			  			     "bad ClassType specified." );
	      }
      }
      
	return *(reinterpret_cast<const PolymorphMap*>( InfoPtrFunc() ) );
  }

  void Model::recordUninitializedStepper( StepperPtr aStepperPtr )
  {
    setDirtyBit();
    uninitializedSteppers.push_back( aStepperPtr );
  }

  void Model::recordUninitializedVariable( VariablePtr aVariablePtr )
  {
    setDirtyBit();
    uninitializedVariables.push_back( aVariablePtr );
  }

  void Model::recordUninitializedSystem( SystemPtr aSystemPtr )
  {
    setDirtyBit();
    uninitializedSystems.push_back( aSystemPtr );
  }

  void Model::recordUninitializedProcess( ProcessPtr aProcessPtr )
  {
    setDirtyBit();
    uninitializedProcesses.push_back( aProcessPtr );
  }

  void Model::createEntity( StringCref aClassname,
			    FullIDCref aFullID )
  {
    if( aFullID.getSystemPath().empty() )
      {
	THROW_EXCEPTION( BadSystemPath, "Empty SystemPath." );
      }

    SystemPtr aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );

    ProcessPtr   aProcessPtr( NULLPTR );
    SystemPtr    aSystemPtr ( NULLPTR );
    VariablePtr aVariablePtr( NULLPTR );

    switch( aFullID.getEntityType() )
      {
      case EntityType::VARIABLE:
	aVariablePtr = getVariableMaker().make( aClassname );
	aVariablePtr->setID( aFullID.getID() );
        aVariablePtr->setCreationTime( getCurrentTime() );
	aContainerSystemPtr->registerVariable( aVariablePtr );

        recordUninitializedVariable( aVariablePtr );

	break;
      case EntityType::PROCESS:
	aProcessPtr = getProcessMaker().make( aClassname );
	aProcessPtr->setID( aFullID.getID() );
	aContainerSystemPtr->registerProcess( aProcessPtr );

        recordUninitializedProcess( aProcessPtr );

	break;
      case EntityType::SYSTEM:
	aSystemPtr = getSystemMaker().make( aClassname );
	aSystemPtr->setID( aFullID.getID() );
	aSystemPtr->setModel( this );
	aContainerSystemPtr->registerSystem( aSystemPtr );

        recordUninitializedSystem( aSystemPtr );

	break;

      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );
      }

    return;
  }

  void Model::removeEntity( FullIDCref aFullID)
  {

    // Put the entity in a list of things to be deleted....
    switch (aFullID.getEntityType() )
      {
      case EntityType::VARIABLE:
        // Check to make sure it isn't found..
        if ( std::find( flaggedVariables.begin(),
                        flaggedVariables.end(),
                        aFullID ) == flaggedVariables.end() )
          {
            flaggedVariables.push_back( aFullID );

            // Find all the dependant processes by scanning through all the systems.

            FullIDVector dependantProcesses;
            SystemPtr rootSystemPtr = getRootSystem();
            recordProcessesDependentOnVariable( rootSystemPtr, aFullID, dependantProcesses);
            
            for(FullIDVectorIterator i = dependantProcesses.begin();
                i != dependantProcesses.end();
                ++i)
              {
                removeEntity( *i );
              }
          }

        break;
        
      case EntityType::PROCESS:
        if (std::find( flaggedProcesses.begin(),
                       flaggedProcesses.end(),
                       aFullID ) == flaggedProcesses.end() )
          {
            flaggedProcesses.push_back( aFullID );
          }
        break;

      case EntityType::SYSTEM:
        if (std::find( flaggedSystems.begin(),
                       flaggedSystems.end(),
                       aFullID ) == flaggedSystems.end() )
          {
            
            // Get a pointer to the system.
            // Call remove contents.  

            SystemPtr parentSystem = getSystem( aFullID.getSystemPath() );
            SystemPtr aSystem = parentSystem->getSystem( aFullID.getID() );
            aSystem->removeContents();

            flaggedSystems.push_back( aFullID );
          }
        break;

      default:
       THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );
      } 
  }


  void Model::eliminateAllFlagged()
  {
    // Iterate over flagged Processes, calling 
    // removeProcess( aFullID );

    for(FullIDVector::iterator i = flaggedProcesses.begin();
        i != flaggedProcesses.end();
        ++i)
      {
        this->removeProcess( *i );

      }

    // Iterate over flagged Variables, removing them one by one.
    // remove Variable.

    for(FullIDVector::iterator i = flaggedVariables.begin();
        i != flaggedVariables.end();
        ++i)
      {
        this->removeVariable( *i );
      }

    // Iterate over systems, removing them one by one.
    // removeSystem( aFullID );

    for(FullIDVector::iterator i = flaggedSystems.begin();
        i != flaggedSystems.end();
        ++i)
      {
        this->removeSystem( *i );
      }
    
    flaggedProcesses.clear();
    flaggedVariables.clear();
    flaggedSystems.clear();
  }
    

  void Model::recordProcessesDependentOnVariable( SystemPtr aSystem, FullID aVariableID, FullIDVector& refDependantProcessesVector)
  {
    for (ProcessMapConstIterator processMapIter = aSystem->getProcessMap().begin();
         processMapIter != aSystem->getProcessMap().end();
         ++processMapIter)
      {
        FullID currentProcessFullID = processMapIter->second->getFullID();
        ProcessPtr currentProcessPtr = processMapIter->second;

        for( VariableReferenceVectorConstIterator variableReferenceIter = currentProcessPtr->getVariableReferenceVector().begin();
             variableReferenceIter != currentProcessPtr->getVariableReferenceVector().end();
             ++variableReferenceIter)
          {
            if (variableReferenceIter->getVariable()->getFullID() == aVariableID )
              {
                refDependantProcessesVector.push_back( currentProcessFullID );
                break;
              }
          }
        
      }

    for( SystemMapConstIterator i = aSystem->getSystemMap().begin() ;
	 i != aSystem->getSystemMap().end() ; 
         ++i )
      {
        // Check things recursively.
	recordProcessesDependentOnVariable( i->second, aVariableID, refDependantProcessesVector);
      }
  }


  void Model::removeVariable( FullIDCref aFullID)
  {
    // We are now *assuming* that nothing that depends on this system
    // exists in the model.  We should assert this somehow....

    SystemPtr aSystem ( getSystem( aFullID.getSystemPath() ) );
    VariablePtr aVariable ( aSystem->getVariable( aFullID.getID() ) );

    aSystem->deleteVariable( aVariable );
    initialize();
  }
  
  void Model::removeProcess( FullIDCref aFullID )
  {
    // We are now *assuming* that nothing that depends on this process
    // exists in the model.  We should assert this somehow....

    SystemPtr aSystem ( getSystem( aFullID.getSystemPath() ) );
    ProcessPtr aProcess ( aSystem->getProcess( aFullID.getID() ) );

    // This apparently does not have to be done here, as deleting the pointer
    // (which is done in aSystem->deleteProcess) appears to remove the process 
    // from it's stepper.
    // 
    // Remove the Process from the Stepper. 
    // aProcess->getStepper()->removeProcess( aProcess );

    // Remove the Process from the system.
    aSystem->deleteProcess( aProcess );
    
    // Reinitialize.
    initialize();
  }

  void Model::removeSystem( FullIDCref aFullID )
  {
    // We are now *assuming* that nothing that depends on this system
    // exists in the model.  We should assert this somehow....

    SystemPtr parentSystem ( getSystem( aFullID.getSystemPath() ) );
    SystemPtr aSystem ( parentSystem->getSystem( aFullID.getID() ) );

    // This is unnecessary I think.
    // Remove the system from it's stepper.
    // aSystem->getStepper()->removeSystem( aSystem );
    
    // Delete the system.
    parentSystem->deleteSystem( aSystem );
    
    initialize();
  }

  SystemPtr Model::getSystem( SystemPathCref aSystemPath ) const
  {
    SystemPath aSystemPathCopy( aSystemPath );

    // 1. "" (empty) means Model itself, which is invalid for this method.
    // 2. Not absolute is invalid (not absolute implies not empty).
    if( ( ! aSystemPathCopy.isAbsolute() ) || aSystemPathCopy.empty() )
      {
	THROW_EXCEPTION( BadSystemPath, 
			 "[" + aSystemPath.getString() +
			 "] is not an absolute SystemPath." );
      }

    aSystemPathCopy.pop_front();

    return getRootSystem()->getSystem( aSystemPathCopy );
  }


  EntityPtr Model::getEntity( FullIDCref aFullID ) const
  {
    EntityPtr anEntity( NULL );
    SystemPathCref aSystemPath( aFullID.getSystemPath() );
    StringCref     anID( aFullID.getID() );

    if( aSystemPath.empty() )
      {
	if( anID == "/" )
	  {
	    return getRootSystem();
	  }
	else
	  {
	    THROW_EXCEPTION( BadID, 
			     "[" + aFullID.getString()
			     + "] is an invalid FullID" );
	  }
      }

    SystemPtr aSystem ( getSystem( aSystemPath ) );

    switch( aFullID.getEntityType() )
      {
      case EntityType::VARIABLE:
	anEntity = aSystem->getVariable( aFullID.getID() );
	break;
      case EntityType::PROCESS:
	anEntity = aSystem->getProcess(  aFullID.getID() );
	break;
      case EntityType::SYSTEM:
	anEntity = aSystem->getSystem(   aFullID.getID() );
	break;
      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );
      }

    return anEntity;
  }


  StepperPtr Model::getStepper( StringCref anID ) const
  {
    StepperMapConstIterator i( theStepperMap.find( anID ) );

    if( i == theStepperMap.end() )
      {
	THROW_EXCEPTION( NotFound, 
			 "Stepper [" + anID + "] not found in this model." );
      }

    return (*i).second;
  }


  void Model::createStepper( StringCref aClassName, StringCref anID )
  {
    StepperPtr aStepper( getStepperMaker().make( aClassName ) );
    aStepper->setModel( this );
    aStepper->setID( anID );

    theStepperMap.insert( std::make_pair( anID, aStepper ) );

    theScheduler.
      addEvent( StepperEvent( getCurrentTime() + aStepper->getStepInterval(),
			      aStepper ) );
  }


  void Model::checkStepper( SystemCptr const aSystem ) const
  {
    if( aSystem->getStepper() == NULLPTR )
      {
        
        THROW_EXCEPTION( InitializationFailed,
                         "No stepper is connected with [" +
                         aSystem->getFullID().getString() + "]." );
      }

    for( SystemMapConstIterator i( aSystem->getSystemMap().begin() ) ;
	 i != aSystem->getSystemMap().end() ; ++i )
      {
	// check it recursively
	checkStepper( i->second );
      }
  }


  void Model::initializeSystems( SystemPtr const aSystem )
  {
    aSystem->initialize();

    for( SystemMapConstIterator i( aSystem->getSystemMap().begin() );
	 i != aSystem->getSystemMap().end() ; ++i )
      {
	// initialize recursively
	initializeSystems( i->second );
      }
  }

  void Model::checkRootSystemSizeVariable()
  {
    // Changed from 
    // void Model::checkSizeVariable( SystemCptr const aSystem )
    // to void Model::checkRootSystemSize()
    // in order to eliminate potential confusion.

    FullID aRootSizeFullID( "Variable:/:SIZE" );

    try
      {
	IGNORE_RETURN getEntity( aRootSizeFullID );
      }
    catch( NotFoundCref )
      {
	createEntity( "Variable", aRootSizeFullID );
	EntityPtr aRootSizeVariable( getEntity( aRootSizeFullID ) );

	aRootSizeVariable->setProperty( "Value", Polymorph( 1.0 ) );
      }
  }

  void Model::initialize()
  {

    // This is added
    if ( getRunningFlag() )
      {
        // Should this be kept?
        for(SystemVector::iterator i = uninitializedSystems.begin();
            i != uninitializedSystems.end();
            ++i)
          {
            (*i)->configureStepper();
          }
      }

    SystemPtr aRootSystem( getRootSystem() );

    checkRootSystemSizeVariable();
    checkStepper( aRootSystem );
    
    initializeSystems( aRootSystem );


    // initialization of Stepper needs four stages:
    // (1) update current times of all the steppers, and integrate Variables.
    // (2) call user-initialization methods of Processes.
    // (3) call user-defined initialize() methods.
    // (4) post-initialize() procedures:
    //     - construct stepper dependency graph and
    //     - fill theIntegratedVariableVector.
    
    FOR_ALL_SECOND( StepperMap, theStepperMap, initializeProcesses );
    FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );

    theSystemStepper.initialize();


    FOR_ALL_SECOND( StepperMap, theStepperMap, 
                    updateIntegratedVariableVector );

    theScheduler.updateEventDependency();
    //    theScheduler.updateAllEvents( getCurrentTime() );

    for( EventIndex c( 0 ); c != theScheduler.getSize(); ++c )
      {
	theScheduler.getEvent(c).reschedule();
      }

    // This is added...
    clearGlobalDirtyState();

    return;
  }

} // namespace libecs





/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
