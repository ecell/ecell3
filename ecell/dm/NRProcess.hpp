#ifndef __NRPROCESS_HPP
#define __NRPROCESS_HPP

#include <limits>
#include <vector>

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/PropertySlotMaker.hpp>
#include <libecs/FullID.hpp>

USE_LIBECS;

DECLARE_CLASS( NRProcess );
DECLARE_VECTOR( NRProcessPtr, NRProcessVector );


/***************************************************************************
     NRProcess 
***************************************************************************/
class NRProcess 
  : 
  public Process
{

  LIBECS_DM_OBJECT( Process, NRProcess );

public:


  NRProcess() 
    :
    theOrder( 0 ),
    theStepInterval( 0.0 ),
    k( 0.0 ),
    Index( -1 )
  {
    CREATE_PROPERTYSLOT_SET_GET( Real, k,            NRProcess );
    CREATE_PROPERTYSLOT_GET    ( Real, Mu,           NRProcess );
    CREATE_PROPERTYSLOT_GET    ( Real, StepInterval, NRProcess );
    CREATE_PROPERTYSLOT_GET    ( Int,  Order,        NRProcess );
      
    CREATE_PROPERTYSLOT( Polymorph, EffectList,
			 NULLPTR,
			 &NRProcess::getEffectListProperty );
  }


  virtual ~NRProcess()
  {
    ; // do nothing
  }



  SIMPLE_SET_GET_METHOD( Real, k );
  SIMPLE_SET_GET_METHOD( Int, Index );


  GET_METHOD( Real, Mu )
  {
    return k * getMultiplicity();
  }


  // The order of the reaction, i.e. 1 for a unimolecular reaction.

  GET_METHOD( Int, Order )
  {
    return theOrder;
  }

  static const Real roundValue( RealCref aValue )
  {
    if( aValue >= 0.0 )
      {
	return floor( aValue );
      }
    else if( aValue > -1.0 ) // if it is -1.0 < > 0.0 round to zero. 
      {                      // this is necessary in hybrid simulation.
	return 0.0;
      }
    else // <= -1.0
      {
	THROW_EXCEPTION( SimulationError, "Variable value <= -1.0" );
      }
  }


  GET_METHOD( Real, Multiplicity )
  {
    Real aMultiplicity( roundValue( theVariableReferenceVector[0].
				    getValue() ) );

    if( getOrder() == 1 )   // one substrate, first order.
      {
	; // do nothing
      }
    else if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
      {  
	aMultiplicity *= roundValue( theVariableReferenceVector[1].
				     getValue() );
      }
    else // one substrate, second order (coeff == -2)
      {
	aMultiplicity *= ( aMultiplicity - 1.0 );
      }

    // this method never return a negative number
    return aMultiplicity;
  }


  GET_METHOD( Real, Volume );
  GET_METHOD( Real, u );

  GET_METHOD( Real, StepInterval )
  {
    return theStepInterval;
  }
    

  GET_METHOD( Real, MinValue )
  {
    // coefficents == 1 only!!! 
      
    VariableReferenceVectorConstIterator
      s( theVariableReferenceVector.begin() );

    Real aMinValue( s->getValue() );
    ++s;

    while( s != theVariableReferenceVector.end() )
      {
	VariableReference aVariableReference( *s );
	const Real aValue( aVariableReference.getValue() );

	if( aValue < aMinValue )
	  {
	    aMinValue = aValue;
	  }

	++s;
      } 

    return aMinValue;
  }

  const Polymorph getEffectListProperty() const;

  
  // a uniform random number (0...1) must be given as u
  void updateStepInterval( const Real u )
  {
    const Real aMu( getMu() );

    // this if is unnecessary: getMultiplicity() always gives >= 0.0
    //if( aMu < 0.0 )
    //      {
    //	THROW_EXCEPTION( SimulationError, "Negative Mu value." );
    //      }

    if( aMu > 0.0 )
      {
	theStepInterval = - log( u ) / aMu;

	if( getOrder() == 2 )
	  {
	    theStepInterval *= getSuperSystem()->getVolume() * N_A;
	  }
      }
    else // aMu == 0.0 (or aMu < 0.0 but this won't happen)
      {
	theStepInterval = std::numeric_limits<Real>::max();
      }
  }



  void calculateOrder();


  virtual void initialize()
  {
    Process::initialize();
    declareUnidirectional();

    calculateOrder();

    if( ! ( getOrder() == 1 || getOrder() == 2 ) )
      {
	THROW_EXCEPTION( ValueError, 
			 String( getClassName() ) + 
			 "[" + getFullID().getString() + 
			 "]: Either first or second order reaction is allowed." );
      }


  }


      
  virtual void process()
  {
    for( VariableReferenceVectorConstIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end() ; ++i )
      {
	VariableReferenceCref aVariableReference( *i );
	aVariableReference.addValue( aVariableReference.getCoefficient() );
      }
  }

  void clearEffectList()
  {
    theEffectList.clear();
  }

  void addEffect( NRProcessPtr anIndex );

  const bool checkEffect( NRProcessPtr anNRProcessPtr ) const;

  NRProcessVectorCref getEffectList() const
  {
    return theEffectList;
  }


protected:

  NRProcessVector theEffectList;

  Int theOrder;

  Real theStepInterval;

  Real k;    
  Int Index;

};


const Polymorph NRProcess::getEffectListProperty() const
{
  PolymorphVector aVector;
  aVector.reserve( theEffectList.size() );

  for ( NRProcessVectorConstIterator i( theEffectList.begin() );
	i != theEffectList.end(); ++i ) 
    {
      NRProcessPtr anNRProcess( *i );

      FullIDCref aFullID( anNRProcess->getFullID() );
      const String aFullIDString( aFullID.getString() );

      aVector.push_back( aFullIDString );
    }

  return aVector;
}


void NRProcess::calculateOrder()
{
  theOrder = 0;
    
  for( VariableReferenceVectorConstIterator 
	 i( theVariableReferenceVector.begin() );
       i != theVariableReferenceVector.end() ; ++i )
    {
      VariableReferenceCref aVariableReference( *i );
      const Int aCoefficient( aVariableReference.getCoefficient() );
	
      // here assume aCoefficient != 0
      if( aCoefficient == 0 )
	{
	  THROW_EXCEPTION( InitializationFailed,
			   "[" + getFullID().getString() + 
			   "]: Zero stoichiometry is not allowed." );
	}

	
      if( aCoefficient < 0 )
	{
	  // sum the coefficient to get the order of this reaction.
	  theOrder -= aCoefficient; 
	}
    }


  // this is checked in initialize()
  // assert( theOrder == 1 || theOrder == 2 );
}


const bool NRProcess::checkEffect( NRProcessPtr anNRProcessPtr ) const
{
  VariableReferenceVectorCref 
    aVariableReferenceVector( anNRProcessPtr->getVariableReferenceVector() );
    
  for( VariableReferenceVectorConstIterator 
	 i( theVariableReferenceVector.begin() );
       i != theVariableReferenceVector.end() ; ++i )
    {
      VariableReferenceCref aVariableReference1(*i );
	
      VariableCptr const aVariable1( aVariableReference1.getVariable() );

      for( VariableReferenceVectorConstIterator 
	     j( aVariableReferenceVector.begin() );
	   j != aVariableReferenceVector.end(); ++j )
	{
	  VariableReferenceCref aVariableReference2( *j );
	  VariableCptr const aVariable2( aVariableReference2.getVariable() );
	  const Int aCoefficient2( aVariableReference2.getCoefficient() );
	    
	  if( aVariable1 == aVariable2 && aCoefficient2 < 0 )
	    {
	      return true;
	    }
	}
    }

  return false;
}




void NRProcess::addEffect( NRProcessPtr aProcessPtr )
{
  if( std::find( theEffectList.begin(), theEffectList.end(), aProcessPtr ) 
      == theEffectList.end() )
    {
      theEffectList.push_back( aProcessPtr );

      // optimization: sort by memory address
      std::sort( theEffectList.begin(), theEffectList.end() );
    }

}

#endif /* __NRPROCESS_HPP */
