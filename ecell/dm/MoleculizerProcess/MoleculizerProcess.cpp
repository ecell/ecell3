#include "MoleculizerProcess.hpp"

#include <libecs/Model.hpp>
#include <boost/lexical_cast.hpp>
#include <set>

#include "utl/dom.hh"


#include "mzr/moleculizer.hh"
#include "mzr/unitsMgr.hh"

#include "CallbackVariable.hpp"


LIBECS_DM_INIT( MoleculizerProcess, Process);

MoleculizerProcess::MoleculizerProcess()
  :
  moleculizerObject( NULL ),
  debugModeFlag( false ),
  Model( ""),
  ModelFile( "" ),
  rxnNdx( 0 ),
  theRegisterClass(new RegisterClass),
  onceInitialized( false )
{}

MoleculizerProcess::~MoleculizerProcess()
{
  delete moleculizerObject;
  delete theRegisterClass;
}



void 
MoleculizerProcess::initialize()
{
  if (!onceInitialized)
    {
      ptrModel = this->getModel();
      compartmentName = this->getFullID().getSystemPath();
      compartmentPtr = this->getModel()->getSystem( compartmentName );

      initializeMoleculizerObject();
      createSpeciesAndReactions(false);

      onceInitialized = true;
    }
  
  Process::initialize();
}


void MoleculizerProcess::fire()
{
  updateSpeciesChanges();
  createSpeciesAndReactions();
  getModel()->initialize();
}

void MoleculizerProcess::initializeMoleculizerObject()
{
  // This appears to be a non-sequiter at the moment.  The reason for this
  // is to try to make this idempotent.  If initialize gets called multiple times
  // we would otherwise have a memory leak.
  delete moleculizerObject;

  String fileName = getModelFile();
  
  xmlpp::DomParser parser;

  parser.set_validate( false );
  parser.parse_file( this->getModelFile() );
  
  mzr::moleculizer* ptrMoleculizerObject = 
    new mzr::moleculizer( parser.get_document() );
  
  this->moleculizerObject = ptrMoleculizerObject;

  std::set<mzr::reaction*> aSet;
  this->moleculizerObject->pUserUnits->theMzrUnit.getMolarFactor().updateVolume(1.0e-18, aSet);
}

void MoleculizerProcess::updateSpeciesChanges()
{

  for(std::vector<std::string>::iterator i = (*theRegisterClass).theVect.begin();
      i != (*theRegisterClass).theVect.end();
      ++i)
    {
      std::set<mzr::reaction*> affectedReactions;

      mzr::species* theSpecies = moleculizerObject->theGeneratedDifferences.generatedSpeciesByName[ *i ];
      theSpecies-> update(affectedReactions, 3);
    }

  theRegisterClass->clearAll();
}

void MoleculizerProcess::createSpeciesAndReactions(bool initPopulationToZero)
{
  mzr::generatedDifference& genDiff = moleculizerObject->theGeneratedDifferences;
  
  for(std::list< mzr::generatedDifference::newSpeciesEntry>::const_iterator iter = genDiff.generatedSpeciesDiff.begin();
      iter != genDiff.generatedSpeciesDiff.end();
      ++iter)
    {
      this->createNewSpecies(*iter, initPopulationToZero);
    }

  //  getModel()->initialize();

  for(std::list< mzr::generatedDifference::newReactionEntry>::const_iterator iter = genDiff.generatedRxnsDiff.begin();
      iter != genDiff.generatedRxnsDiff.end();
      ++iter)
    {
      this->createNewReaction(*iter);
    }

  // getModel()->initialize();

  genDiff.clearAll();
  return;
}
  
void MoleculizerProcess::createNewSpecies(const newSpeciesEntry& newSpecies, bool initPopulationToZero)
{
  const mzr::species* const ptrNewSpecies(newSpecies.first);
  String newSpeciesName(newSpecies.second.second);
  SpeciesKey newSpeciesKey = newSpecies.second.first;
  Integer newSpeciesPopulation;

  FullID newSpeciesFullID( EntityType("Variable"),
                           this->compartmentName,
                           newSpeciesName );

  if (initPopulationToZero) 
    {
      newSpeciesPopulation = 0;
    }
  else 
    {
      newSpeciesPopulation = newSpecies.first->getPop();
    }
  
  ptrModel->createEntity("CallbackVariable", newSpeciesFullID);
  EntityPtr newVariable = ptrModel->getEntity( newSpeciesFullID );

  CallbackVariable* newCallbackVariable = dynamic_cast<CallbackVariable*>(newVariable);
  newCallbackVariable->registerCallback( this->theRegisterClass );
  
  ptrModel->getEntity( newSpeciesFullID )->setProperty("Value", newSpeciesPopulation);
  return;
}

void MoleculizerProcess::createNewReaction(const newReactionEntry& newRxn)
{

  const mzr::reaction* const ptrMzrGeneratedRxn( newRxn.first );
  String newReactionName( newRxn.second.second );
  ReactionKey newReactionKey = newRxn.second.first;

  FullID newRxnFullID( EntityType("Process"),
                       this->compartmentName,
                       "Rxn_" + boost::lexical_cast<String>(this->rxnNdx++));

  ptrModel->createEntity("GillespieProcess", newRxnFullID);

  EntityPtr newEntityPtr = ptrModel->getEntity( newRxnFullID );
  ProcessPtr newRxnPtr = dynamic_cast<ProcessPtr>(newEntityPtr);


  newRxnPtr->setProperty("k", ptrMzrGeneratedRxn->getRate() );
  newRxnPtr->setStepperID( this->GillespieProcessStepperID );

  const std::map<mzr::species*, int>& rxnSubstrates( ptrMzrGeneratedRxn->getSubstrateMap() );
  const std::map<mzr::species*, int>& rxnProducts( ptrMzrGeneratedRxn->getProductMap() );

  std::for_each( rxnSubstrates.begin(),
                 rxnSubstrates.end(),
                 addSubstratesToRxn( newRxnPtr, this->compartmentPtr ));

  std::for_each( rxnProducts.begin(),
                 rxnProducts.end(),
                 addProductsToRxn( newRxnPtr, this->compartmentPtr, *this));

  return;
}


void
MoleculizerProcess::addSubstratesToRxn::operator()(std::pair<mzr::species*, int> aSubstrate)
{
  // This should already exist...
  String substrateName( aSubstrate.first->getCanonicalName() );
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
MoleculizerProcess::addProductsToRxn::operator()(std::pair<mzr::species*, int> aProduct)
{

  String productName( aProduct.first->getCanonicalName() );
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
