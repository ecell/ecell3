//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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


#include "libecs/libecs.hpp"
#include "libecs/Message.hpp"
#include "libecs/FQPI.hpp"
#include "libecs/RootSystem.hpp"
#include "libecs/SubstanceMaker.hpp"
#include "libecs/ReactorMaker.hpp"
#include "libecs/SystemMaker.hpp"
#include "libecs/Stepper.hpp"

#include "LocalSimulatorImplementation.hpp"

namespace libemc
{

  using namespace libecs;

  LocalSimulatorImplementation::LocalSimulatorImplementation()
    :
    theRootSystem( *new RootSystem )
  {
    ; // do nothing
  }

  LocalSimulatorImplementation::~LocalSimulatorImplementation()
  {
    delete &theRootSystem;
  }

  void LocalSimulatorImplementation::createEntity( StringCref classname,
						   FQPICref fqpi, 
						   StringCref name )
  {
    PrimitiveType aType = fqpi.getPrimitiveType();
    SystemPtr aTargetSystem = getRootSystem().getSystem( fqpi );

    SubstancePtr aSubstancePtr;
    ReactorPtr   aReactorPtr;
    SystemPtr    aSystemPtr;

    switch( aType )
      {

      case SUBSTANCE:
	aSubstancePtr = getRootSystem().getSubstanceMaker().make( classname );
	aSubstancePtr->setId( fqpi.getIdString() );
	aSubstancePtr->setName( name );
	aTargetSystem->registerSubstance( aSubstancePtr );
	break;

      case REACTOR:
	aReactorPtr = getRootSystem().getReactorMaker().make( classname );
	aReactorPtr->setId( fqpi.getIdString() );
	aReactorPtr->setName( name );
	aTargetSystem->registerReactor( aReactorPtr );
	break;

      case SYSTEM:
	aSystemPtr = getRootSystem().getSystemMaker().make( classname );
	aSystemPtr->setId( fqpi.getIdString() );
	aSystemPtr->setName( name );
	aTargetSystem->registerSystem( aSystemPtr );
	break;

      case NONE:
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );

      }

  }

  void LocalSimulatorImplementation::setProperty( FQPICref fqpi, 
						  MessageCref message)
  {
    PrimitiveType aType = fqpi.getPrimitiveType();
    SystemPtr aSystem = getRootSystem().getSystem( SystemPath(fqpi) );

    EntityPtr anEntityPtr;

    switch( aType )
      {

      case SUBSTANCE:
	anEntityPtr = aSystem->getSubstance( fqpi.getIdString() );
	break;

      case REACTOR:
	anEntityPtr = aSystem->getReactor( fqpi.getIdString() );
	break;

      case SYSTEM:
	anEntityPtr = aSystem->getSystem( fqpi.getIdString() );
	break;

      case NONE:
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );

      }

    anEntityPtr->set( message );
  }


  const Message LocalSimulatorImplementation::
  getProperty( FQPICref fqpi,
	       StringCref propertyName )
  {
    PrimitiveType aType = fqpi.getPrimitiveType();
    SystemPtr aSystem = getRootSystem().getSystem( fqpi );

    EntityPtr anEntityPtr;

    switch( aType )
      {

      case SUBSTANCE:
	anEntityPtr = aSystem->getSubstance( fqpi.getIdString() );
	break;

      case REACTOR:
	anEntityPtr = aSystem->getReactor( fqpi.getIdString() );
	break;

      case SYSTEM:
	anEntityPtr = aSystem->getSystem( fqpi.getIdString() );
	break;

      case NONE:
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );

      }

    return anEntityPtr->get( propertyName );
  }


  void LocalSimulatorImplementation::step()
  {
    theRootSystem.getStepperLeader().step();  
  }

  void LocalSimulatorImplementation::initialize()
  {
    theRootSystem.initialize();
  }



} // namespace libemc

