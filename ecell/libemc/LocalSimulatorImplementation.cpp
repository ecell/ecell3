#include "Message.hpp"
#include "FQPI.hpp"

#include "LocalSimulatorImplementation.hpp"

LocalSimulatorImplementation::LocalSimulatorImplementation()
{
  ; // do nothing
}

void LocalSimulatorImplementation::makePrimitive( StringCref classname,
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
      aTargetSystem->addSubstance( aSubstancePtr );
      break;
    case REACTOR:
      aReactorPtr = getRootSystem().getReactorMaker().make( classname );
      aReactorPtr->setId( fqpi.getIdString() );
      aReactorPtr->setName( name );
      aTargetSystem->addReactor( aReactorPtr );
      break;
    case SYSTEM:
      aSystemPtr = getRootSystem().getSystemMaker().make( classname );
      aSystemPtr->setId( fqpi.getIdString() );
      aSystemPtr->setName( name );
      aTargetSystem->addSystem( aSystemPtr );
      break;
    case NONE:
    default:
      throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				  "bad PrimitiveType specified." );
    }

}

void LocalSimulatorImplementation::sendMessage( FQPICref fqpi, 
						MessageCref message)
{
  PrimitiveType aType = fqpi.getPrimitiveType();
  SystemPtr aSystem = getRootSystem().getSystem( SystemPath(fqpi) );

  cerr << aSystem->getId() << endl;

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


Message LocalSimulatorImplementation::getMessage( FQPICref fqpi,
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



