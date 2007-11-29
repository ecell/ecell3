
#include "libecs.hpp"
#include "Process.hpp"
#include "Model.hpp"
#include "System.hpp"
#include <iostream>
#include <sstream>

USE_LIBECS;

LIBECS_DM_CLASS( DivisionProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( DivisionProcess, Process )
    {
      INHERIT_PROPERTIES( Process );
      
      PROPERTYSLOT_SET_GET( Real, VolumeAtDivision );
    }

  DivisionProcess()
    :
    firecount(0)
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, VolumeAtDivision );

  virtual void initialize()
    {
      Process::initialize();
    }

  virtual void fire()
    {
      Real VolumeAtDivision = getVolumeAtDivision();

      SystemPtr aSysPtr = getSuperSystem();

      Real size1 = aSysPtr->getSize();
      Real size2 = getSuperSystem()->findSizeVariable()->getValue();
      
      //      getModel()->getRootSystem()->printSystems()

      if ( size1 >= VolumeAtDivision )
        {
          divideCell();
        }

      return;

    }

 protected:

  void divideCell()
  {
    std::ostringstream i;
    i << numberOfCells++;
    String cellName( "CELL" );
    cellName += i.str();

    std::cout << getSuperSystem()->getID() << " is dividing into " << this->getSuperSystem()->getID() << " and " << cellName << std::endl;

    createNewCellWithName(cellName );
    getModel()->initialize();
    divideCellContentsInto( cellName );
    getModel()->initialize();
    createCellularProcesses( cellName );
    getModel()->initialize();

    return;
  }

  void createNewCellWithName( StringCref cellName )
  {
  
    ModelPtr theModel = getModel();
    theModel->createEntity( "System", FullID( EntityType( "System"),
                                              "/", 
                                              cellName));
  }

  void divideCellContentsInto( StringCref newCellName ) 
  {
    // This should create the A, B, and SIZE variables...
    
    VariableMapCref theVariableMap( getSuperSystem()->getVariableMap() );
    std::map<const String, VariablePtr>::const_iterator anIter;
    
    for(anIter = theVariableMap.begin();
        anIter != theVariableMap.end();
        ++anIter)
      {
        String entityName( anIter->first );
        VariablePtr theOldVariable = anIter->second;
        VariablePtr newVariablePtr;



        Real oldValue = theOldVariable->getValue();
        Real newValue = oldValue / 2.0;

        // Create a new variable with the same name but in the new cell.
        FullID aFullID( EntityType( "Variable" ), 
                        "/" + newCellName,
                        entityName);

        getModel()->createEntity( "Variable", aFullID);
        newVariablePtr = getModel()->getSystem( aFullID.getSystemPath() )->getVariable( aFullID.getID() );
        
        // Update the values of both old and new entities.
        theOldVariable->setValue( newValue );
        newVariablePtr->setValue( newValue );
      }
  }

  void createCellularProcesses( StringCref newCellName )
  {
    createCellGrowthProcess( newCellName );
    createA_SynthesisProcess( newCellName );
    createAToBConversionProcess( newCellName );
    createCellDivisionProcess( newCellName );
  }


  void createCellGrowthProcess(StringCref newCellName)
  {
    // Create the cell growth process.....
    FullID newCellGrowthProcessFullID( EntityType("Process"),
                                       "/" + newCellName,
                                       "CellGrowthProcess_" + newCellName);

    getModel()->createEntity("ExpressionFluxProcess", newCellGrowthProcessFullID);
    ProcessPtr theNewCellGrowthProcess( getModel()->getSystem( SystemPath("/" + newCellName) )->getProcess("CellGrowthProcess_" + newCellName) );

    theNewCellGrowthProcess->registerVariableReference("P", 
                                                       getModel()->getSystem( SystemPath("/" + newCellName) )->getVariable("SIZE"),
                                                       1, true);

    theNewCellGrowthProcess->setProperty("Expression", Polymorph("(1.57e-11 / 2.0) * (3 * P.Value / (4 * 3.14159) )^(1.0/3.0)") );
    //theNewCellGrowthProcess->setProperty("Expression", Polymorph("1e-15") );
  }

  void createA_SynthesisProcess( StringCref newCellName )
  {
    // Create the A_SynthesisProcess

    FullID newASynthProcessFullID( EntityType("Process"),
                                   "/" + newCellName,
                                   "A_SynthesisProcess");
    getModel()->createEntity("ExpressionFluxProcess", newASynthProcessFullID);
    ProcessPtr theNewASynthProcessPtr( getModel()->getSystem( SystemPath("/" + newCellName))->getProcess("A_SynthesisProcess") );
    theNewASynthProcessPtr->registerVariableReference("P", getModel()->getSystem( SystemPath("/" + newCellName) )->getVariable("A"), 1, true);
    theNewASynthProcessPtr->setProperty("Expression", Polymorph("10.0") );
  }

  void createAToBConversionProcess( StringCref newCellName )
  {
    // Create the A_Degredation Process.


    FullID newADegrProcessFullID(EntityType("Process"),
                                 "/" + newCellName,
                                 "A_DegradationProcess");

    getModel()->createEntity("MassActionFluxProcess", newADegrProcessFullID);
    ProcessPtr ADegradationProcessPtr( getModel()->getSystem( SystemPath("/" + newCellName))->getProcess("A_DegradationProcess") );

    ADegradationProcessPtr->setProperty("k", 0.11);
    ADegradationProcessPtr->registerVariableReference("P", getModel()->getSystem( SystemPath("/" + newCellName) )->getVariable("A"), -1, true);
    ADegradationProcessPtr->registerVariableReference("S", getModel()->getSystem( SystemPath("/" + newCellName) )->getVariable("B"), 1, true);

    return;
  }


  void createCellDivisionProcess( StringCref newCellName )
  {

    FullID newCellularDivisionProcess( EntityType("Process"),
                                       "/" + newCellName,
                                         "DivisionProcess");
      
      getModel()->createEntity("DivisionProcess", newCellularDivisionProcess);
      ProcessPtr divisionProcessPtr( divisionProcessPtr = getModel()->getSystem( SystemPath("/" + newCellName) )->getProcess("DivisionProcess") );

      divisionProcessPtr->setProperty("VolumeAtDivision", 4e-15);
      divisionProcessPtr->setStepper( this->getStepper() );

      SystemPtr parentSystem = getModel()->getSystem( SystemPath("/" + newCellName) );
      VariablePtr ptrA = parentSystem->getVariable("A");
      VariablePtr ptrB = parentSystem->getVariable("B");
      VariablePtr ptrSIZE = parentSystem->getVariable("SIZE");
        
      divisionProcessPtr->registerVariableReference("S1", ptrA, 1, true);      
      divisionProcessPtr->registerVariableReference("S2", ptrB, 1, true);
      divisionProcessPtr->registerVariableReference("S3", ptrSIZE, 1, true);

      return;
  }


  Real VolumeAtDivision;
  static Integer numberOfCells;
  int firecount;
  
};

Integer DivisionProcess::numberOfCells = 1;
LIBECS_DM_INIT( DivisionProcess, Process );


