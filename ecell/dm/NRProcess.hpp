#ifndef __NRPROCESS_HPP
#define __NRPROCESS_HPP

#include <limits>
#include <vector>

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/PropertySlotMaker.hpp>
#include <libecs/FullID.hpp>

#include "NRStepper.hpp"

USE_LIBECS;

//DECLARE_CLASS( NRProcess );
//DECLARE_VECTOR( NRProcessPtr, NRProcessVector );


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


    GET_METHOD( Real, Multiplicity )
    {
      Real aMultiplicity( 1.0 );

      for( VariableReferenceVectorConstIterator 
	     s( theVariableReferenceVector.begin() );
	   s != theZeroVariableReferenceIterator ; ++s )
	{
	  VariableReference aVariableReference(*s );
          const Int aCoefficient( abs( aVariableReference.getCoefficient() ) );
          const Real aValue( floor( aVariableReference.getValue() ) );

	  //FIXME: what if the value is negative?


   	  for( UnsignedInt i( 0 ); i < aCoefficient; ++i ) 
	  {
            aMultiplicity *= aValue;
          }
        }

      //      std::cerr << getID() << ": multiplicity: " << aMultiplicity << "\n"<< std::endl;



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


    void updateStepInterval()
    {
      const Real aMu( getMu() );

      if( aMu == 0.0 )
	{
	  theStepInterval = std::numeric_limits<Real>::max();
	  return;
	}

      if( aMu < 0.0 )
	{
	  THROW_EXCEPTION( SimulationError, "Negative Mu value." );
	}

      //  ( 0...1 )
      Real u( gsl_rng_uniform_pos( theNRStepper->getRng() ) );

      Real aMuV( aMu );
      if( getOrder() == 2 )
	{
	  aMuV /= getSuperSystem()->getVolume();
	}

      const Real a( - 1 / aMuV );

      theStepInterval =  a * log( u );  // + getCurrentTime()
    }



    void calculateOrder();


    virtual void initialize()
    {
      Process::initialize();

      theNRStepper = dynamic_cast<NRStepperPtr>( getStepper() );
      if( theNRStepper == NULLPTR )
	{
	  THROW_EXCEPTION( InitializationFailed, 
			   "[" + getFullID().getString() + 
			   "]: NRProcess must be used in conjunction with NRStepper" );

	}



      calculateOrder();

      if( ! ( getOrder() == 1 || getOrder() == 2 ) )
	{
	  THROW_EXCEPTION( ValueError, 
			   String( getClassName() ) + 
			   "[" + getFullID().getString() + 
			   "]: Either first or second order reaction is allowed." );
	}

      updateEffectList();


    }


      
    virtual void process()
    {
      //      std::cerr << getID();

      for( VariableReferenceVectorConstIterator 
	     i( theVariableReferenceVector.begin() );
	   i != theVariableReferenceVector.end() ; ++i )
	{
	  VariableReferenceCref aVariableReference( *i );
	
	  const Int aCoefficient( aVariableReference.getCoefficient() );

	  Real aValue( aVariableReference.getValue() + aCoefficient );

	  //	  std::cerr << ' ' << aVariableReference.getVariable()->getID() << ' ' << aValue;
	  
	  aVariableReference.setValue( aValue );
	}

      //      std::cerr << std::endl;

    }


    void addEffect( NRProcessPtr anIndex );

    void updateEffectList();

    const bool checkEffect( NRProcessPtr anNRProcessPtr ) const;

    NRProcessVectorCref getEffectList() const
    {
      return theEffectList;
    }


  protected:

    NRProcessVector theEffectList;

    Int theOrder;
    NRStepperPtr    theNRStepper;

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

    assert( theOrder == 1 || theOrder == 2 );

  }


  void NRProcess::updateEffectList()
  {
    theEffectList.clear();

    // here assume aCoefficient != 0

    NRProcessVectorCref 
      anNRProcessVector( theNRStepper->getNRProcessVector() );
    for( NRProcessVectorConstIterator i( anNRProcessVector.begin() );
	 i != anNRProcessVector.end(); ++i )
      {
	NRProcessPtr const anNRProcessPtr( *i );

	if( checkEffect( anNRProcessPtr ) )
	  {
	    addEffect( anNRProcessPtr );
	  }
      }

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
      }

  }

#endif /* __NRPROCESS_HPP */
