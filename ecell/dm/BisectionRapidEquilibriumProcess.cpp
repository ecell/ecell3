// written by kem

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "FluxProcess.hpp"
#include "ecell3_dm.hpp"

#include <iostream>
#include <string>

#define ECELL3_DM_TYPE Process

// macros for debugging
#define COUT_VAR( NAME )\
std::cout << # NAME << " :" << std::endl;\
std::cout << "\t" << NAME << std::endl//

#define COUT( LINE )\
std::cout << LINE << std::endl//

#define CIN()\
Int a;\
std::cin >> a//
// macro for debugging

// macro for loops
#define EACH_SUBSTRATE( ITER )\
VariableReferenceVectorIterator ITER( theVariableReferenceVector.begin() );\
ITER != theZeroVariableReferenceIterator;\
ITER++//

#define EACH_PRODUCT( ITER )\
VariableReferenceVectorIterator ITER( thePositiveVariableReferenceIterator );\
ITER != theVariableReferenceVector.end();\
ITER++//

#define EACH_VARIABLE_REFERENCE( ITER )\
VariableReferenceVectorIterator ITER( theVariableReferenceVector.begin() );\
ITER != theVariableReferenceVector.end();\
ITER++//
// macro for loops

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{
  ECELL3_DM_OBJECT;
  
 protected:
  
  Real Keq;
  Real KeqConc;
  String UnitKeq;
  
  Real Tolerance;
  Real ToleranceConc;
  String UnitTolerance;
  
  Real theVelocityBuffer;
  
 public:
  
  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Keq );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, Tolerance );
      
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, KeqConc );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, ToleranceConc );
      
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( String, UnitKeq );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( String, UnitTolerance );
      
      theVelocityBuffer = 0;
    }
  
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, Tolerance );
  
  SIMPLE_SET_GET_METHOD( Real, KeqConc );
  SIMPLE_SET_GET_METHOD( Real, ToleranceConc );
  
  SIMPLE_SET_GET_METHOD( String, UnitKeq );
  SIMPLE_SET_GET_METHOD( String, UnitTolerance );
  
  virtual void initialize()
    {
      FluxProcess::initialize();
      
      // set KeqConc
      Int aDimention( 0 );
      if( UnitKeq == String( "molecules" ) )
	{
	  for( EACH_VARIABLE_REFERENCE( anIter ) )
	    {
	      aDimention -= (*anIter).getCoefficient();
	    }
	  Real aVolume = getSuperSystem()->getVolume();
	  KeqConc = Keq * pow( ( N_A * aVolume ) , aDimention );
	}
      else
	{
	  KeqConc = Keq;
	}
      
      // set ToleranceConc
      if( Tolerance == 0 ) // to set 0 in em-file means "use default value".
	{
	  ToleranceConc = convertUnitToConcentration( 0.1 );
	}
      else if( UnitTolerance == String( "molecules" ) )
	{
	  ToleranceConc = convertUnitToConcentration( Tolerance );
	}
      else
	{
	  ToleranceConc = Tolerance;
	}
    }
  
  // sum( substrate ) * Keq - sum( product )
  Real getDiff( Real aVelocity )
    {
      Real aLeftSide( KeqConc );
      for( EACH_SUBSTRATE( aSubstrateIter ) )
	{
	  Real aConc( (*aSubstrateIter).getConcentration() );
	  Int aCoeff( -1 * (*aSubstrateIter).getCoefficient() );
	  aConc -= aCoeff * aVelocity;
	  aLeftSide *= pow( aConc , aCoeff );
	}

      Real aRightSide( 1 );
      for( EACH_PRODUCT( aProductIter ) )
	{
	  Real aConc( (*aProductIter).getConcentration() );
	  Int aCoeff( (*aProductIter).getCoefficient() );
	  aConc += aCoeff * aVelocity;
	  aRightSide *= pow( aConc , aCoeff );
	}

      return aLeftSide - aRightSide;
    }

  VariableReferencePtr getMinSubstratePtr()
    {
      VariableReferencePtr aMinSubstratePtr( &(theVariableReferenceVector[0]));
      Real aConc = theVariableReferenceVector[ 0 ].getConcentration();
      Real aCoeff = theVariableReferenceVector[ 0 ].getCoefficient();
      Real aMin( aConc / aCoeff );
      for( EACH_SUBSTRATE( aSubstrateIter ) )
	{
	  Real aConc = (*aSubstrateIter).getConcentration();
	  Real aCoeff = (*aSubstrateIter).getCoefficient();
	  if( aMin <  aConc / aCoeff )
	    aMin = aConc / aCoeff;
	  aMinSubstratePtr = &(*aSubstrateIter);
	}
      return aMinSubstratePtr;
    }
  
  VariableReferencePtr getMinProductPtr()
    {
      Int anIndex = getPositiveVariableReferenceOffset();
	VariableReferencePtr
	aMinProductPtr( &theVariableReferenceVector[ anIndex ] );
      
      Real aConc = theVariableReferenceVector[ anIndex ].getConcentration();
      Real aCoeff = theVariableReferenceVector[ anIndex ].getCoefficient();
      Real aMin( aConc / aCoeff );
      for( EACH_PRODUCT( aProductIter ) )
	{
	  Real aConc = (*aProductIter).getConcentration();
	  Real aCoeff = (*aProductIter).getCoefficient();
	  if( aMin >  aConc / aCoeff )
	    aMin = aConc / aCoeff;
	  aMinProductPtr = &(*aProductIter);
	}
      return aMinProductPtr;
    }

  Real getInitialVelocity()
    {
      Real aDiff = getDiff( 0 );
      if( aDiff == 0 )
	{
	  return 0;
	}
      else if( aDiff > 0 ) // velocity is positive number
	{
	  if( theVelocityBuffer > 0 )
	    {
	      return theVelocityBuffer;
	    }
	  else // opposite sign or first turn( theVelocityBuffer == 0 ).
	    {
	      VariableReferencePtr aMinSubstratePtr = getMinSubstratePtr();
	      Real aConc = aMinSubstratePtr->getConcentration();
	      Real aCoeff = aMinSubstratePtr->getCoefficient();
	      return -1 * aConc / ( aCoeff * 2 );
	    }
	}
      else                 // velocity is negative number
	{
	  if( theVelocityBuffer < 0 )
	    {
	      return theVelocityBuffer;
	    }
	  else // opposite sign or first turn( theVelocityBuffer == 0 ).
	    {
	      VariableReferencePtr aMinProductPtr = getMinProductPtr();
	      Real aConc = aMinProductPtr->getConcentration();
	      Real aCoeff = aMinProductPtr->getCoefficient();
	      return -1 * aConc / ( aCoeff * 2 );
	    }
	}
    }
  
  Real optimizeVelocity( Real aVelocity , Real aWidth , Real aDiffBuffer )
    {
      if( -1 * ToleranceConc < aWidth && aWidth < ToleranceConc )
	{
	  theVelocityBuffer = aVelocity;
	  return aVelocity;
	}

      Real aDiff = getDiff( aVelocity );
      if( aDiff * aDiffBuffer < 0 )
	{
	  aWidth *= -1;
	}
      
      aVelocity += aWidth;
      aWidth /= 2;
      return optimizeVelocity( aVelocity, aWidth , aDiff );
    }
  
  virtual void process()
    {
      Real aVelocity = getInitialVelocity();

      while( ( getDiff( 0 ) * getDiff( aVelocity ) ) > 0 )
	{
	  aVelocity *= 2;
	}

      aVelocity = optimizeVelocity( aVelocity,
				    -1 * aVelocity / 2, 
				    getDiff( aVelocity ) );
      aVelocity = convertUnitToQty( aVelocity );
      
      for( EACH_VARIABLE_REFERENCE( anIter ) )
	{
	  VariableReferencePtr aVariableReferencePtr( &(*anIter) );
	  Real aValue( aVariableReferencePtr->getValue() );
	  Int aCoeff( aVariableReferencePtr->getCoefficient() );
	  aVariableReferencePtr->setValue( aValue + ( aCoeff * aVelocity ) );
	}
    }


  Real convertUnitToQty( Real aConcentration )
    {
      return  aConcentration * N_A * getSuperSystem()->getVolume();
    }
  
  Real convertUnitToConcentration( Real aQty )
    {
      return  aQty / ( N_A * getSuperSystem()->getVolume() );
    }
  
};

ECELL3_DM_INIT;
