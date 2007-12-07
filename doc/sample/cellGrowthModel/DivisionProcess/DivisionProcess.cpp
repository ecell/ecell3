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
// Contact information:
//   Nathan Addy, Research Associate     Voice: 510-981-8748
//   The Molecular Sciences Institute    Email: addy@molsci.org  
//   2168 Shattuck Ave.                  
//   Berkeley, CA 94704
//
//END_HEADER

#include <iostream>
#include <sstream>
#include <boost/assign/std/vector.hpp>

#include "DivisionProcess.hpp"
using namespace std;

USE_LIBECS;

LIBECS_DM_INIT( DivisionProcess, Process );

void DivisionProcess::initialize()
{
  PythonProcessBase::initialize();
  
  python::handle<> a( PyEval_EvalCode( (PyCodeObject*)
				       theCompiledInitializeMethod.ptr(),
				       theGlobalNamespace.ptr(), 
				       theLocalNamespace.ptr() ) );
}

void DivisionProcess::fire()
{
  python::object a( python::eval( theExpression.c_str(),
				  theGlobalNamespace, 
				  theLocalNamespace ) );
  
  bool cellDivisionCondition = python::extract<bool>(a)();
      
  if ( cellDivisionCondition ) 
    {
      divideCell();
    }
  
  return;
}

void DivisionProcess::divideCell()
{

  // Step 1: halve the values (recursively of all the variables in the system)
  halveSystemVariables( getSuperSystem() );

  // Step 2: Create a new (sibling) system for the new cell.
  String newCellName = this->getNextCellName();
  SystemPtr newCellSystem = createNewCellWithName( newCellName );

  // Copy over the subsystem structure.
  copySubsystemsToSystem( getSuperSystem(), newCellSystem );
  
  // Copy over the variables
  copyVariablesToSystem( getSuperSystem(), newCellSystem);

  // Copy over the processes
  copyProcessesToSystem( getSuperSystem(), newCellSystem);

  return;
}

void DivisionProcess::copyProcessesToSystem( SystemPtr fromSystem, SystemPtr toSystem )
{
  SystemPath destinationSystemPath( toSystem->getSystemPath() );
  destinationSystemPath.push_back( toSystem->getID() );
  
  for( ProcessMapConstIterator processIter = getSuperSystem()->getProcessMap().begin();
       processIter != getSuperSystem()->getProcessMap().end();
       ++processIter)
    {
      // Create a new copy of this type of process in the new system
      FullID newProcessFullID( EntityType::PROCESS,
                               destinationSystemPath,
                               processIter->first
                               );

      getModel()->createEntity( processIter->second->getClassName(),
                                newProcessFullID) ;
     
      StringVector dontCopyTheseProperties;
      dontCopyTheseProperties.push_back( "VariableReferenceList" );

      // dontCopyTheseProperties += "VariableReferenceList", "Activity", "MolarActivity";

      

      ProcessPtr originalProcess( fromSystem->getProcess( processIter->first ) );
      ProcessPtr copyProcess( toSystem->getProcess( processIter->first ) );

      copyVariableReferenceListPropertyBetweenProcesses( originalProcess, copyProcess);
      copyPropertiesBetweenEntities(originalProcess, copyProcess, dontCopyTheseProperties);


      getModel()->initialize();      
    }

  // Call this function recursively.
  for(SystemMapConstIterator sysIter = fromSystem->getSystemMap().begin();
      sysIter != fromSystem->getSystemMap().end();
      ++sysIter)
    {
      SystemPtr fromSystemPrime = fromSystem->getSystem( sysIter->first );
      SystemPtr toSystemPrime = toSystem->getSystem( sysIter->first );

      copyProcessesToSystem( fromSystemPrime, toSystemPrime );
    }
}

String DivisionProcess::getNextCellName()
{
  
  std::ostringstream outputStream;
  outputStream << ++numberOfCells;
  
  return ( "Cell_" + outputStream.str() );
}


SystemPtr DivisionProcess::createNewCellWithName( StringCref cellName )
{
  FullID newCellFullID( EntityType::SYSTEM,
                        getSuperSystem()->getFullID().getSystemPath(), 
                        cellName);
  
  getModel()->createEntity( "System",
                            newCellFullID );
  getModel()->initialize();
  
  SystemPtr newSystem = getModel()->getSystem( newCellFullID.getSystemPath() )->getSystem( newCellFullID.getID() );
  return newSystem;
}

void DivisionProcess::halveSystemVariables( SystemPtr aSystemPtr )
{
  // This function takes a SystemPtr as its argument and halves each
  // of the variables in that system as well as in all dependant
  // systems.

  // Halve each of the variables in system 'aSystemPtr'.
  VariableMapCref systemVariableMap( aSystemPtr->getVariableMap() );

  for(VariableMapConstIterator varIter = systemVariableMap.begin();
      varIter != systemVariableMap.end();
      ++varIter)
    {
      VariablePtr nonConstVariablePtr = aSystemPtr->getVariable( varIter->second->getID() );

      // This works ok if currentValue is a Real...
      // What if it is an Int?
      Real halfCurrentValue = nonConstVariablePtr->getValue()/2.0f;
      nonConstVariablePtr->setValue( halfCurrentValue );
    }

  // Call the function on each of the subsystems recursively.
  SystemMapCref systemSystemMap( aSystemPtr->getSystemMap() );

  for(SystemMapConstIterator sysIter = systemSystemMap.begin();
      sysIter != systemSystemMap.end();
      ++sysIter)
    {
      SystemPtr nonConstSystem = aSystemPtr->getSystem( sysIter->second->getID() );
      halveSystemVariables( nonConstSystem );
    }
}

void DivisionProcess::copySubsystemsToSystem( SystemPtr fromSystem, SystemPtr toSystem)
{
  // First we construct the system path that the new systems will be added to.
  SystemPath newSystemPath( toSystem->getFullID().getSystemPath() );
  newSystemPath.push_back( toSystem->getFullID().getID() );

  for(SystemMapConstIterator sysIter = fromSystem->getSystemMap().begin();
      sysIter != fromSystem->getSystemMap().end();
      ++sysIter)
    {
      FullID newFullID( EntityType::SYSTEM,
                        newSystemPath,
                        sysIter->first );

      getModel()->createEntity( "System", newFullID );
      getModel()->initialize();

      SystemPtr fromSystemPrime = fromSystem->getSystem( sysIter->first );
      SystemPtr toSystemPrime = toSystem->getSystem( sysIter->first );

      copySubsystemsToSystem( fromSystemPrime, toSystemPrime);
    }
}

void DivisionProcess::copyVariablesToSystem( SystemPtr fromSystem, SystemPtr toSystem)
{

  SystemPath newSystemPath( toSystem->getFullID().getSystemPath() );
  newSystemPath.push_back( toSystem->getID() );

  // Here, where I am using "old" model structure to create new model structure
  // (creating new model objects by iterating through the original model contents)
  // I should make a copy of "getVariableMap" to ensure the map I am iterating in
  // doesn't get messed up by adding objects.  This critique applies to all member
  // functions of this type.

  for(VariableMapConstIterator varIter = fromSystem->getVariableMap().begin();
      varIter != fromSystem->getVariableMap().end();
      ++varIter)
    {

      // Create a new variable with the same ID as *varIter, but in the new system.
      FullID newVariableFullID( EntityType::VARIABLE,
                                newSystemPath,
                                varIter->first );

      getModel()->createEntity(varIter->second->getClassName(), newVariableFullID );
      getModel()->initialize();

      // Set the value to the value of the pointed to variable.
      // Optimally we would get the PropertyList and copy those over appropriately, one by one.
      VariablePtr newVariablePtr( getModel()->getSystem( newVariableFullID.getSystemPath() )->getVariable( newVariableFullID.getID() ) );
      newVariablePtr->setValue( varIter->second->getValue() );

      StringVector dontCopyTheseProperties;
      //      dontCopyTheseProperties += "DiffusionCoeff", "MolarConc", "NumberConc", "Velocity";

      copyPropertiesBetweenEntities( varIter->second, 
                                     newVariablePtr, 
                                     dontCopyTheseProperties );
    }


  // Call this function recursively.
  for(SystemMapConstIterator sysIter = fromSystem->getSystemMap().begin();
      sysIter != fromSystem->getSystemMap().end();
      ++sysIter)
    {
      SystemPtr fromSystemPrime = fromSystem->getSystem( sysIter->first );
      SystemPtr toSystemPrime = toSystem->getSystem( sysIter->first );

      copyVariablesToSystem( fromSystemPrime, toSystemPrime );
    }

}

void DivisionProcess::copyPropertiesBetweenEntities( const EntityPtr fromEntity, EntityPtr toEntity, StringVectorCref propertiesNotToCopy)
{
  // Iterate through all of the properties in fromEntity.  If they are copyable and do not appear in 
  // "propertiesNotToCopy", copy it.

  PolymorphVector thePropertyList = fromEntity->getPropertyList().asPolymorphVector();
  
  // For each property of the Entity...
  for( PolymorphVectorConstIterator iter = thePropertyList.begin();
       iter != thePropertyList.end();
       ++iter)
    {

      String propertyName = iter->asString();

      // If the property isn't excluded...
      if ( std::find( propertiesNotToCopy.begin(), 
                      propertiesNotToCopy.end(),
                      propertyName ) == propertiesNotToCopy.end() )
        {
          

          PolymorphVector thePropertySlotProperties( fromEntity->getPropertyAttributes( propertyName ).asPolymorphVector() );
          bool isSettable( thePropertySlotProperties[0].asInteger() );
          bool isGettable( thePropertySlotProperties[1].asInteger() );

          // And if the property is gettable and settable...
          if ( isSettable && isGettable )
            {

              // Then copy the property value of the first to the property slot of the second...

              Polymorph toProperty = fromEntity->getProperty( propertyName );
              toEntity->setProperty( propertyName, toProperty );
            }
        }

      
    }

}


void DivisionProcess::copyVariableReferenceListPropertyBetweenProcesses( const ProcessPtr fromProcess, ProcessPtr toProcess)
{
  assert (fromProcess->getProperty("VariableReferenceList").getType() == Polymorph::POLYMORPH_VECTOR );

  PolymorphVector variableReferenceList = fromProcess->getProperty( "VariableReferenceList" ).asPolymorphVector();

  for(PolymorphVectorIterator i = variableReferenceList.begin();
      i != variableReferenceList.end();
      ++i)
    {
      assert( i->getType() == Polymorph::POLYMORPH_VECTOR );
      PolymorphVector variableReference( i->asPolymorphVector() );

      Polymorph variableReferenceName( variableReference[0] );
      Polymorph variableFullIDPolymorph( variableReference[1] );
      Polymorph variableCoefficient( variableReference[2] );
      Polymorph variableIsAccessor( variableReference[3] );

      
      FullID fromProcessVarReferenceFullID( "Variable" + variableFullIDPolymorph.asString() );

      SystemPath fromProcessSystemPath = fromProcess->getFullID().getSystemPath();
      SystemPath variableReferenceSystemPath = fromProcessVarReferenceFullID.getSystemPath();
      

      // Create a SystemPath S such that FullID( "Variable", fromProcessSystemPath + S, ID) == FullID( Variable, variableReferenceSystemPath )

      SystemPath relativePath;
      
      StringListIterator varRefSystemPathIterator = variableReferenceSystemPath.begin();
      StringListIterator processSystemPathIterator = fromProcessSystemPath.begin();
      
      while( *processSystemPathIterator == *varRefSystemPathIterator )
        {
          processSystemPathIterator++;
          varRefSystemPathIterator++;

          if (processSystemPathIterator == fromProcessSystemPath.end() || varRefSystemPathIterator == variableReferenceSystemPath.end() )
            {
              break;
            }
        }
      
      while( processSystemPathIterator++ != fromProcessSystemPath.end() )
        {
          relativePath.push_back( String("..") );
        }

      while(varRefSystemPathIterator != variableReferenceSystemPath.end() )
        {
          relativePath.push_back( *varRefSystemPathIterator++);
        }
      
      SystemPath newProcessSystemPath( toProcess->getFullID().getSystemPath() );

      for(StringListIterator relativePathIter = relativePath.begin();
          relativePathIter != relativePath.end();
          ++relativePathIter)
        {
          newProcessSystemPath.push_back( *relativePathIter );
        }

      FullID newFullID( EntityType::VARIABLE,
                        newProcessSystemPath,
                        fromProcessVarReferenceFullID.getID() );

      VariablePtr theVarPtr = getModel()->getSystem( newFullID.getSystemPath() )->getVariable( newFullID.getID() );
      assert( theVarPtr != 0 );
      
      toProcess->registerVariableReference(variableReferenceName.asString(),
                                           theVarPtr,
                                           variableCoefficient.asInteger(),
                                           variableIsAccessor.asInteger());
      
    }
  
}

Integer DivisionProcess::numberOfCells = 0;



