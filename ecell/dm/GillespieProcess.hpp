#ifndef __GILLESPIEPROCESS_HPP
#define __GILLESPIEPROCESS_HPP

#include <limits>
#include <vector>

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/PropertySlotMaker.hpp>
#include <libecs/FullID.hpp>

USE_LIBECS;

DECLARE_CLASS( GillespieProcess );
DECLARE_VECTOR( GillespieProcessPtr, GillespieProcessVector );


/***************************************************************************
     GillespieProcess 
***************************************************************************/
class GillespieProcess 
  : 
  public Process
{

  LIBECS_DM_OBJECT( Process, GillespieProcess );

  typedef const Real (GillespieProcess::* RealMethodPtr)() const;

public:


  GillespieProcess() 
    :
    theOrder( 0 ),
    theStepInterval( 0.0 ),
    k( 0.0 ),
    Index( -1 ),
    theGetMultiplicityMethodPtr( &GillespieProcess::getZero ),
    theGetMinValueMethodPtr( &GillespieProcess::getZero )
  {
    CREATE_PROPERTYSLOT_SET_GET( Real, k,            GillespieProcess );
    CREATE_PROPERTYSLOT_GET    ( Real, Mu,           GillespieProcess );
    CREATE_PROPERTYSLOT_GET    ( Real, StepInterval, GillespieProcess );
    CREATE_PROPERTYSLOT_GET    ( Int,  Order,        GillespieProcess );
      
    CREATE_PROPERTYSLOT( Polymorph, EffectList,
			 NULLPTR,
			 &GillespieProcess::getEffectListProperty );
  }


  virtual ~GillespieProcess()
  {
    ; // do nothing
  }


  SIMPLE_SET_GET_METHOD( Real, k );
  SIMPLE_SET_GET_METHOD( Int, Index );


  GET_METHOD( Real, Mu )
  {
    return k * ( this->*theGetMultiplicityMethodPtr )();
  }


  // The order of the reaction, i.e. 1 for a unimolecular reaction.

  GET_METHOD( Int, Order )
  {
    return theOrder;
  }

  GET_METHOD( Real, StepInterval )
  {
    return theStepInterval;
  }

  GET_METHOD( Real, MinValue )
  {
    return ( this->*theGetMinValueMethodPtr )();
  }

  const Polymorph getEffectListProperty() const;

  
  // a uniform random number (0...1) must be given as u
  void updateStepInterval( const Real u )
  {
    const Real aMu( getMu() );

    if( aMu > 0.0 )
      {
	theStepInterval = - log( u ) / aMu;

	if( getOrder() == 2 )
	  {
	    theStepInterval *= getSuperSystem()->getVolumeN_A();
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
			 "]: Only first or second order scheme is allowed." );
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

  void addEffect( GillespieProcessPtr anIndex );

  const bool checkEffect( GillespieProcessPtr anGillespieProcessPtr ) const;

  GillespieProcessVectorCref getEffectList() const
  {
    return theEffectList;
  }


protected:


  inline static const Real roundValue( RealCref aValue )
  {
    const Real aRoundedValue( trunc( aValue ) );

    if( aRoundedValue < 0.0 )
      {
	THROW_EXCEPTION( SimulationError, "Variable value <= -1.0" );
      }

    return aRoundedValue;
  }


  const Real getZero() const
  {
    return 0.0;
  }

  const Real getMultiplicity_FirstOrder() const
  {
    return roundValue( theVariableReferenceVector[0].getValue() );
  }

  const Real getMultiplicity_SecondOrder_TwoSubstrates() const
  {
    Real aMultiplicity( roundValue( theVariableReferenceVector[0].
				    getValue() ) );
    aMultiplicity *= roundValue( theVariableReferenceVector[1].getValue() );

    return aMultiplicity;
  }

  const Real getMultiplicity_SecondOrder_OneSubstrate() const
  {
    Real aMultiplicity( roundValue( theVariableReferenceVector[0].
				    getValue() ) );

    aMultiplicity *= ( aMultiplicity - 1.0 ) * 0.5;

    return aMultiplicity;
  }

  const Real getMinValue_FirstOrder() const
  {
    return theVariableReferenceVector[0].getValue();
  }

  const Real getMinValue_SecondOrder_TwoSubstrates() const
  {
    const Real aFirstValue( theVariableReferenceVector[0].getValue() );
    const Real aSecondValue( theVariableReferenceVector[1].getValue() );

    if( aFirstValue < aSecondValue )
      {
	return aFirstValue;
      }
    else
      {
	return aSecondValue;
      }
  }

  const Real getMinValue_SecondOrder_OneSubstrate() const
  {
    return theVariableReferenceVector[0].getValue() * 0.5;
  }


protected:

  GillespieProcessVector theEffectList;

  Int theOrder;

  Real theStepInterval;

  Real k;    
  Int Index;

  RealMethodPtr theGetMultiplicityMethodPtr;
  RealMethodPtr theGetMinValueMethodPtr;

};


const Polymorph GillespieProcess::getEffectListProperty() const
{
  PolymorphVector aVector;
  aVector.reserve( theEffectList.size() );

  for ( GillespieProcessVectorConstIterator i( theEffectList.begin() );
	i != theEffectList.end(); ++i ) 
    {
      GillespieProcessPtr anGillespieProcess( *i );

      FullIDCref aFullID( anGillespieProcess->getFullID() );
      const String aFullIDString( aFullID.getString() );

      aVector.push_back( aFullIDString );
    }

  return aVector;
}


void GillespieProcess::calculateOrder()
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

  // set theGetMultiplicityMethodPtr and theGetMinValueMethodPtr

  if( getOrder() == 0 )   // no substrate
    {
      theGetMultiplicityMethodPtr = &GillespieProcess::getZero;
      theGetMinValueMethodPtr     = &GillespieProcess::getZero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      theGetMultiplicityMethodPtr = 
	&GillespieProcess::getMultiplicity_FirstOrder;
      theGetMinValueMethodPtr = &GillespieProcess::getMinValue_FirstOrder;
    }
  else if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
    {  
      theGetMultiplicityMethodPtr = 
	&GillespieProcess::getMultiplicity_SecondOrder_TwoSubstrates;
      theGetMinValueMethodPtr = 
	&GillespieProcess::getMinValue_SecondOrder_TwoSubstrates;
    }
  else // one substrate, second order (coeff == -2)
    {
      theGetMultiplicityMethodPtr = 
	&GillespieProcess::getMultiplicity_SecondOrder_OneSubstrate;
      theGetMinValueMethodPtr = 
	&GillespieProcess::getMinValue_SecondOrder_OneSubstrate;
    }

}


const bool GillespieProcess::checkEffect( GillespieProcessPtr 
					  anGillespieProcessPtr ) const
{
  VariableReferenceVectorCref 
    aVariableReferenceVector( anGillespieProcessPtr->
			      getVariableReferenceVector() );
    
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




void GillespieProcess::addEffect( GillespieProcessPtr aProcessPtr )
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
