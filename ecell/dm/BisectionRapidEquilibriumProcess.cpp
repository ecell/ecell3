// written by kem

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "Process.hpp"

#include <iostream>
#include <string>

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

LIBECS_DM_CLASS( BisectionRapidEquilibriumProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( BisectionRapidEquilibriumProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, Tolerance );

      PROPERTYSLOT_SET_GET( Real, KeqConc );
      PROPERTYSLOT_SET_GET( Real, ToleranceConc );
      
      PROPERTYSLOT_SET_GET( String, UnitKeq );
      PROPERTYSLOT_SET_GET( String, UnitTolerance );
    } 

  
 protected:
  
  Real Keq;
  Real KeqConc;
  String UnitKeq;
  
  Real Tolerance;
  Real ToleranceConc;
  String UnitTolerance;
  
  Real theVelocityBuffer;
  
 public:
  
  BisectionRapidEquilibriumProcess()
    {
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
      Process::initialize();
      
      // set KeqConc
      Int aDimention( 0 );
      if( UnitKeq == String( "molecules" ) )
	{
	  for( EACH_VARIABLE_REFERENCE( anIter ) )
	    {
	      aDimention -= (*anIter).getCoefficient();
	    }
	  Real aSize = getSuperSystem()->getSize();
	  KeqConc = Keq * pow( ( N_A * aSize ) , aDimention );
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
	  Real aConc( (*aSubstrateIter).getMolarConc() );
	  Int aCoeff( -1 * (*aSubstrateIter).getCoefficient() );
	  aConc -= aCoeff * aVelocity;
	  aLeftSide *= pow( aConc , aCoeff );
	}

      Real aRightSide( 1 );
      for( EACH_PRODUCT( aProductIter ) )
	{
	  Real aConc( (*aProductIter).getMolarConc() );
	  Int aCoeff( (*aProductIter).getCoefficient() );
	  aConc += aCoeff * aVelocity;
	  aRightSide *= pow( aConc , aCoeff );
	}

      return aLeftSide - aRightSide;
    }

  VariableReferencePtr getMinSubstratePtr()
    {
      VariableReferencePtr aMinSubstratePtr( &(theVariableReferenceVector[0]));
      Real aConc = theVariableReferenceVector[ 0 ].getMolarConc();
      Real aCoeff = theVariableReferenceVector[ 0 ].getCoefficient();
      Real aMin( aConc / aCoeff );
      for( EACH_SUBSTRATE( aSubstrateIter ) )
	{
	  Real aConc = (*aSubstrateIter).getMolarConc();
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
      
      Real aConc = theVariableReferenceVector[ anIndex ].getMolarConc();
      Real aCoeff = theVariableReferenceVector[ anIndex ].getCoefficient();
      Real aMin( aConc / aCoeff );
      for( EACH_PRODUCT( aProductIter ) )
	{
	  Real aConc = (*aProductIter).getMolarConc();
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
	      Real aConc = aMinSubstratePtr->getMolarConc();
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
	      Real aConc = aMinProductPtr->getMolarConc();
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
      return  aConcentration * N_A * getSuperSystem()->getSize();
    }
  
  Real convertUnitToConcentration( Real aQty )
    {
      return  aQty / ( N_A * getSuperSystem()->getSize() );
    }
  
};

LIBECS_DM_INIT( BisectionRapidEquilibriumProcess, Process );
