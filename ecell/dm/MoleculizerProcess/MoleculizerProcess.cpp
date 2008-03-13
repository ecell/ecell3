#include "MoleculizerProcess.hpp"

#include <libecs/Model.hpp>
#include <boost/lexical_cast.hpp>
#include <set>

#include "utl/dom.hh"

#include "mzr/moleculizer.hh"
#include "mzr/unitsMgr.hh"
#include "mzr/mzrUnit.hh"

#include "CallbackVariable.hpp"


LIBECS_DM_INIT( MoleculizerProcess, Process);

MoleculizerProcess::MoleculizerProcess()
  :
  moleculizerObject( new mzr::moleculizer ),
  debugMode( false ),
  ModelDescription( ""),
  ModelFile( "" ),
  rxnNdx( 0 ),
  theRegisterClass(new RegisterClass),
  onceInitialized( false ),
  NetworkExpansionDepth( 1 ),
  rateExtrapolation( true )
{}

MoleculizerProcess::~MoleculizerProcess()
{
  delete moleculizerObject;
  delete theRegisterClass;
}

void 
MoleculizerProcess::initialize()
{
  if ( !onceInitialized )
    {
      // Only do this the very first time this function is called.  
      onceInitialized = true;

      ptrModel = this->getModel();

      compartmentPath = this->getFullID().getSystemPath();
      compartmentPtr = this->getModel()->getSystem( compartmentPath );

      // Create the moleculizer* and then add the species and reactions to the model.
      initializeMoleculizerObject();
      createSpeciesAndReactions(false);
    }
  
  // This is the part that constantly reinitializes.
  Process::initialize();
}


void MoleculizerProcess::fire()
{
  if (getDebug() )
    {
      cout << "Moleculizer::Process::fire()" << endl;
    }

  expandMoleculizerNetworkBySpecies();
  createSpeciesAndReactions();
  getModel()->initialize();
}

void MoleculizerProcess::initializeMoleculizerObject()
{

  if (getDebug() )
    {
      cout << "MoleculizerProcess::initializeMoleculizerObject()" << endl;
    }

  // Set the volume for the libmoleculizer object. 

  if (!compartmentPtr)
    {

      throw UnexpectedError("MoleculizerProcess::initializeMoleculizerObject()",
                            "CompartmentPtr should be set but isn't.");
    }
  
  
  Real theVolume = compartmentPtr->getSize();
  fnd::sensitivityList<mzr::mzrReaction> aSet;  

  moleculizerObject->pUserUnits->pMzrUnit->getMolarFactor().updateVolume(theVolume, aSet);
  // moleculizerObject->pUserUnits->pMzrUnit->getMolarFactor().updateVolume(1e-18, aSet);


  // We should also set the generation depth as well as the RateExtrapolation here.
  moleculizerObject->setGenerateDepth( getNetworkExpansionDepth() );

  // This actually does nothing at the moment.  libmoleculizer should be extended to fix this.
  moleculizerObject->setRateExtrapolation( getRateExtrapolation() );
  
  if ( getModelDescription() != "" )
    {
      // Initialize Moleculizer with the model.
      String model = getModelDescription();
      moleculizerObject->attachString( model );
    }
  else if ( getModelFile() != "" )
    {


      // Initialize Moleculizer with the file
      String fileName = getModelFile();
      moleculizerObject->attachFileName( fileName );
      if (getDebug() )
        {
          cout << "Attaching Modelfile..." << endl;
        }
    }
  else
    {
      // Throw a different, better error.
      throw UnexpectedError("MoleculizerProcess::initializeMoleculizerObject()",
                            "No model information has been set.");
    }

  if (getDebug() )
    {
      cout << "Generated " << moleculizerObject->getNumberDeltaSpecies()  << "species." << endl;
      cout << "Generated " << moleculizerObject->getNumberDeltaReactions()  << "reactions." << endl;
    }
}

void MoleculizerProcess::expandMoleculizerNetworkBySpecies()
{
  if (getDebug() )
    {
      cout << "MoleculizerProcess::expandMoleculizerNetworkBySpecies()" << endl;
    }


  // In this function we iterate through theRegisterClass, which contains the ID's of all
  // the species which have updated for the first time, and tell the moleculizer object 
  // to expand the reaction network by way of these species.

  for(std::vector<std::string>::iterator i = (*theRegisterClass).theVect.begin();
      i != (*theRegisterClass).theVect.end();
      ++i)
    {
      moleculizerObject->incrementNetworkBySpeciesName( *i );
    }
  
  theRegisterClass->clearAll();
}

 void MoleculizerProcess::createSpeciesAndReactions(bool initPopulationToZero)
 {

  if (getDebug() )
    {
      cout << "MoleculizerProcess::createSpeciesAndReactions()" << endl;
    }

   for(std::list<mzr::mzrSpecies*>::const_iterator iter = (*moleculizerObject).theDeltaSpeciesList.begin();
       iter != (*moleculizerObject).theDeltaSpeciesList.end();
       ++iter)
     {
       this->createNewSpecies( (*iter)->getName(), initPopulationToZero);
     }

   cout << "Reinitializing..." << endl;
   getModel()->initialize();

   for(std::list< mzr::mzrReaction*>::const_iterator iter = (*moleculizerObject).theDeltaReactionList.begin();
       iter != (*moleculizerObject).theDeltaReactionList.end();
       ++iter)
     {
       this->createNewReaction(*iter);
     }
   
   cout << "Reinitializing..." << endl;
   getModel()->initialize();
   moleculizerObject->resetCurrentState();

   return;
}
  
void MoleculizerProcess::createNewSpecies(const String& newSpecies, bool initPopulationToZero)
{
  if (getDebug() )
    {
      cout << "MoleculizerProcess::creatingNewSpecies() -- creating\n\t " << newSpecies << endl;
    }
  
  FullID newSpeciesFullID( EntityType("Variable"),
                           this->compartmentPath,
                           newSpecies );

  Integer newSpeciesPopulation(0);
  if (!initPopulationToZero) 
    {
      newSpeciesPopulation = moleculizerObject->getPopulation( newSpecies );
    }
  
  ptrModel->createEntity("CallbackVariable", newSpeciesFullID);
  EntityPtr newVariable = ptrModel->getEntity( newSpeciesFullID );

  // I can totally understand where this comes from, but it's just a little obscene.
  CallbackVariable* newCallbackVariable = dynamic_cast<CallbackVariable*>(newVariable);
  newCallbackVariable->registerCallback( this->theRegisterClass );
  
  ptrModel->getEntity( newSpeciesFullID )->setProperty("Value", newSpeciesPopulation);
  return;
}

void MoleculizerProcess::createNewReaction(const mzr::mzrReaction* newRxn)
{
  
  if (getDebug() )
    {
      cout << "MoleculizerProcess::createNewReaction -- \n\t" << newRxn->getName() << endl;
    }

  FullID newRxnFullID( EntityType("Process"),
                       this->compartmentPath,
                       "Rxn_" + boost::lexical_cast<String>(this->rxnNdx++));

  ptrModel->createEntity("GillespieProcess", newRxnFullID);

  EntityPtr newEntityPtr = ptrModel->getEntity( newRxnFullID );
  ProcessPtr newRxnPtr = dynamic_cast<ProcessPtr>(newEntityPtr);


  newRxnPtr->setProperty("k", newRxn->getRate() );
  newRxnPtr->setStepperID( this->GillespieProcessStepperID );

  const std::map<mzr::mzrSpecies*, int>& rxnSubstrates( newRxn->getReactants() );
  const std::map<mzr::mzrSpecies*, int>& rxnProducts( newRxn->getProducts() );

  std::for_each( rxnSubstrates.begin(),
                 rxnSubstrates.end(),
                 addSubstratesToRxn( newRxnPtr, this->compartmentPtr ));

  std::for_each( rxnProducts.begin(),
                 rxnProducts.end(),
                 addProductsToRxn( newRxnPtr, this->compartmentPtr, *this));

  return;
}


void
MoleculizerProcess::addSubstratesToRxn::operator()(std::pair<mzr::mzrSpecies*, int> aSubstrate)
{
  // This should already exist...
  String substrateName( aSubstrate.first->getName() );
  SystemPath containingSystemPath = rxnProcessPtr->getFullID().getSystemPath();

  // We must check this ptr to make sure it's ok.
  VariablePtr ptrSubstrateVariable = parentSystemPtr->getVariable( substrateName );

  String variableReferenceName(alphabet, substrateNdx, substrateNdx + 1);
  substrateNdx++;

  Integer multiplicity( aSubstrate.second );

  rxnProcessPtr->registerVariableReference( variableReferenceName,
                                            ptrSubstrateVariable,
                                            -1 * multiplicity,
                                            false);

  return;
}

void
MoleculizerProcess::addProductsToRxn::operator()(std::pair<mzr::mzrSpecies*, int> aProduct)
{

  String productName( aProduct.first->getName() );
  SystemPath containingSystemPath = rxnProcessPtr->getFullID().getSystemPath();

  // We must check this ptr to make sure it's ok.
  VariablePtr ptrProductVariable = parentSystemPtr->getVariable( productName );

  String variableReferenceName(alphabet, productNdx, productNdx + 1);
  productNdx++;

  Integer multiplicity( aProduct.second );


  rxnProcessPtr->registerVariableReference( variableReferenceName,
                                            ptrProductVariable,
                                            1 * multiplicity,
                                            false);
  return;
}

const String MoleculizerProcess::addSubstratesToRxn::alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const String MoleculizerProcess::addProductsToRxn::alphabet = "ZYXWVUTSRQPONMLKJIHGFEDCBA";
