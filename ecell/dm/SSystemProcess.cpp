#ifndef __SSYSTEMPROCESS_HPP
#define __SSYSTEMPROCESS_HPP

#include <gsl/gsl_sf.h>
#include <vector>

#include "libecs.hpp"

#include "ESSYNSProcess.hpp"

USE_LIBECS;

DECLARE_VECTOR( Real, RealVector );

LIBECS_DM_CLASS( SSystemProcess, ESSYNSProcess )
{

 public:

  LIBECS_DM_OBJECT( SSystemProcess, Process )
    {
      INHERIT_PROPERTIES( ESSYNSProcess );

      PROPERTYSLOT_SET_GET( Integer, Order );
      PROPERTYSLOT_SET_GET( Polymorph, SSystemMatrix );
    }


  SSystemProcess()
    :
    theSystemSize( 0 ),
    Order( 3 )
    {
      ; // do nothing
    }

  virtual ~SSystemProcess()
    {
      ;
    }

  SIMPLE_GET_METHOD( Integer, Order );
  SET_METHOD( Integer, Order );
  
  SET_METHOD( Polymorph, SSystemMatrix );

  GET_METHOD( Polymorph, SSystemMatrix )
    {
      return SSystemMatrix;
    }

  void fire()
    {
      ;
    }  

  const std::vector<RealVector>& getESSYNSMatrix();
  
  GET_METHOD( Integer, SystemSize )
    {
      return theSystemSize;
    }
 
  void initialize()
    {
      Process::initialize();
    }  
  
   
 protected:

  int Order;
  int theSystemSize;

  Polymorph SSystemMatrix;
  
  // State variables in log space
  std::vector< RealVector > theY;
  
  // S-System vectors
  RealVector theAlpha;
  RealVector theBeta;

  std::vector< RealVector > theG;
  std::vector< RealVector > theH;
  
  // tmp S-System vectors
  std::vector< RealVector > theAlphaBuffer;
  std::vector< RealVector > theBetaBuffer;
  std::vector< RealVector > theGBuffer;
  std::vector< RealVector > theHBuffer;
  std::vector< RealVector > theFBuffer;
  
};

LIBECS_DM_INIT( SSystemProcess, Process );

SET_METHOD_DEF( Integer, Order, SSystemProcess )
{ 
  Order = value;
  
  // init Substance Vector
  theY.resize(theSystemSize+1);
  RealVector tmp;
  tmp.resize(Order+1);
  for(int i( 0 ); i < theSystemSize + 1; i++)
    {
      theY[i] = tmp;
    }

  // init S-System Vector & Matrix
  theAlpha.resize(theSystemSize+1);
  theBeta.resize(theSystemSize+1);
  theG.resize( theSystemSize + 1);
  theH.resize( theSystemSize + 1);
  tmp.resize(theSystemSize+1);
  for(int i( 0 ); i < theSystemSize + 1; i++)
    {
      theG[i] = tmp;
      theH[i] = tmp;
    }

  // init S-System tmp Vector & Matrix
  theAlphaBuffer.resize( theSystemSize + 1);
  theBetaBuffer.resize( theSystemSize + 1);
  theGBuffer.resize( theSystemSize + 1);
  theHBuffer.resize( theSystemSize + 1);
  tmp.resize(Order+1);
  for(int i( 0 ); i < theSystemSize + 1; i++)
    {
      theAlphaBuffer[i] = tmp;
      theBetaBuffer[i] = tmp;
      theGBuffer[i] = tmp;
      theHBuffer[i] = tmp;
    }

  theFBuffer.resize( Order + 1);
  tmp.resize(Order);
  for(int i( 0 ); i < Order + 1; i++)
    {
      theFBuffer[i] = tmp;
    }  

}

SET_METHOD_DEF( Polymorph, SSystemMatrix, SSystemProcess )
{
  SSystemMatrix = value;
  PolymorphVector aValueVector( value.asPolymorphVector() );
  theSystemSize = aValueVector.size();

  // init Substance Vector
  theY.resize(theSystemSize+1);
  RealVector tmp;
  tmp.resize(Order+1);
  for(int i( 0 ); i < theSystemSize + 1; i++)
    {
      theY[i] = tmp;
    }

  // init S-System Vector & Matrix
  theAlpha.resize(theSystemSize+1);
  theBeta.resize(theSystemSize+1);
  theG.resize( theSystemSize + 1);
  theH.resize( theSystemSize + 1);
  tmp.resize(theSystemSize+1);
  for(int i( 0 ); i < theSystemSize + 1; i++)
    {
      theG[i] = tmp;
      theH[i] = tmp;
    }

  // init S-System tmp Vector & Matrix
  theAlphaBuffer.resize( theSystemSize + 1);
  theBetaBuffer.resize( theSystemSize + 1);
  theGBuffer.resize( theSystemSize + 1);
  theHBuffer.resize( theSystemSize + 1);
  tmp.resize(Order+1);

  for(int i( 0 ); i < theSystemSize + 1; i++)
    {
      theAlphaBuffer[i] = tmp;
      theBetaBuffer[i] = tmp;
      theGBuffer[i] = tmp;
      theHBuffer[i] = tmp;
    }

  theFBuffer.resize( Order + 1);
  tmp.resize(Order);
  for(int i( 0 ); i < Order + 1; i++)
    {
      theFBuffer[i] = tmp;
    }  

  // init Factorial matrix
  for(int m( 2 ) ; m < Order+1 ; m++)
    {
      for(int q( 1 ); q < m ; q++)
	{
	  const Real aFact( 1 / gsl_sf_fact(q-1) * gsl_sf_fact(m-q-1) * m * (m-1) );
	  (theFBuffer[m])[q] = aFact;      
	}
    }

  // set Alpha, Beta, G, H 
  for( int i( 0 ); i < theSystemSize; i++ )
    {
      
      theAlpha[i+1] = (aValueVector[i].asPolymorphVector())[0].asReal() ;
      for( int j( 0 ); j < theSystemSize; j++ )	
	{
	  if( i == j )
	    {
	      
	      (theG[i+1])[j+1] = (aValueVector[i].asPolymorphVector())[j+1].asReal() - 1 ;
	    }else{
	      (theG[i+1])[j+1] = (aValueVector[i].asPolymorphVector())[j+1].asReal() ;
	    }
	}
      theBeta[i+1] = (aValueVector[i].asPolymorphVector())[1+theSystemSize].asReal() ;
      for( int j( 0 ); j < theSystemSize; j++ )	
	{
	  if( i == j )
	    {
	      (theH[i+1])[j+1] = (aValueVector[i].asPolymorphVector())[2+j+theSystemSize].asReal() -1 ;
	    }else{
	      (theH[i+1])[j+1] = (aValueVector[i].asPolymorphVector())[2+j+theSystemSize].asReal() ;
	    }
	}
    }
}

const std::vector<RealVector>& SSystemProcess::getESSYNSMatrix()
{
  //get theY
  int anIndex = 0;
  for( VariableReferenceVectorConstIterator
	 i ( thePositiveVariableReferenceIterator );
       i != theVariableReferenceVector.end() ; ++i )
    {
      if( (*i).getVariable()->getValue() <= 0 )
	{
	  THROW_EXCEPTION( ValueError, "Error:in SSystemPProcess::fire().log() in 0.");
	}
      (theY[anIndex])[0] =
	gsl_sf_log( (*i).getVariable()->getValue() ) ;
      anIndex++;
    }
  
  //differentiate first order
  for(int i( 1 ) ; i < theSystemSize+1 ; i++ )
    {
      Real aGt( 0.0 );
      Real aHt( 0.0 );
      for(int j( 1 ) ; j < theSystemSize+1 ; j++ )
	{
	  aGt += (theG[i])[j] * (theY[j-1])[0];
	  aHt += (theH[i])[j] * (theY[j-1])[0];
	}
      
      Real aAlpha = theAlpha[i] * exp(aGt);
      Real aBate  = theBeta[i] * exp(aHt);
      
      (theAlphaBuffer[i])[1] = aAlpha;
      (theBetaBuffer[i])[1] = aBate;
      (theY[i-1])[1] =  aAlpha - aBate;
    }
 
  //differentiate second and/or more order
  for( int m( 2 ) ; m <= Order ; m++)
    {
      for(int i( 1 ) ; i < theSystemSize+1 ; i++ )
	{
	 
	  (theGBuffer[i])[m-1] = 0;
	  (theHBuffer[i])[m-1] = 0;
	 
	  for(int j( 1 ) ; j < theSystemSize+1 ; j++ )
	    {
	      const Real aY( (theY[j-1])[m-1] );
	      const Real aG( (theGBuffer[i])[m-1] );
	      const Real aH( (theHBuffer[i])[m-1] );
	     
	      (theGBuffer[i])[m-1] = 
		aG + (theG[i])[j] * aY ;
	      (theHBuffer[i])[m-1] =
		aH + (theH[i])[j] * aY ;
	    }
	}
     
      for(int i( 1 ) ; i < theSystemSize+1 ; i++ )
	{
	  (theAlphaBuffer[i])[m] = 0;
	  (theBetaBuffer[i])[m] = 0;
	 
	  for(int q( 1 );  0 > m-q ; q++)
	    {
	      (theAlphaBuffer[i])[m] = 
		(theAlphaBuffer[i])[m] + 
		(theFBuffer[m])[q] *
		(theAlphaBuffer[i])[m-q] *
		(theGBuffer[i])[m-q] ;
	      (theBetaBuffer[i])[m] =
		(theBetaBuffer[i])[m]  + 
		(theFBuffer[m])[q] *
		(theBetaBuffer[i])[m-q] *
		(theHBuffer[i])[m-q] ;
	    }
	 
	  (theY[i-1])[m] = 
	    (theAlphaBuffer[i])[m] - 
	    (theBetaBuffer[i])[m] ;
	}
     
    }
  
  return theY;
 
  /*
  //integrate
  for( int i( 1 ); i < theSystemSize+1; i++)
  {
     
  Real aY( 0.0 );
  for( int m( 1 ); m <= Order ; m++)
  {
  aY += (theY[i-1])[m] * 
  gsl_sf_pow_int(aStepInterval,m) / gsl_sf_fact(m);
  }
  (theY[i-1])[0] =  aY + (theY[i-1])[0] ;
  }

  //set value
  anIndex = 0;
  for( VariableReferenceVectorConstIterator
  i ( thePositiveVariableReferenceIterator );
  i != theVariableReferenceVector.end() ; ++i )
  {
  (*i).getVariable()->setValue( exp( (theY[anIndex])[0] ) );
  anIndex++;
  }
  */
 
 
}


#endif /* __SSYSTEMPROCESS_HPP */
