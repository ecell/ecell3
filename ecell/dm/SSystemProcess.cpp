#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_sf.h>

#include "libecs.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"

#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
  : 
  public Process
{

  ECELL3_DM_OBJECT;
  
 public:

  ECELL3_DM_CLASSNAME()
    :
    theSystemSize( 0 ),
    Order( 3 ),
    theChangedFlag( false ),
    theY( NULLPTR ),
    theAlpha( NULLPTR ),
    theBeta( NULLPTR ),
    theG( NULLPTR ),
    theH( NULLPTR ),
    theAlphaBuffer( NULLPTR ),
    theBetaBuffer( NULLPTR ),
    theGBuffer( NULLPTR ),
    theHBuffer( NULLPTR ),
    theFBuffer( NULLPTR )
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Int, Order );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Polymorph, SSystemMatrix );
    }
  
  ~ECELL3_DM_CLASSNAME()
    {
      gsl_vector_free( theAlpha );
      gsl_vector_free( theBeta );
      gsl_matrix_free( theY );
      gsl_matrix_free( theG );
      gsl_matrix_free( theH );
      gsl_matrix_free( theAlphaBuffer );
      gsl_matrix_free( theBetaBuffer );
      gsl_matrix_free( theGBuffer );
      gsl_matrix_free( theHBuffer );
      gsl_matrix_free( theFBuffer );
    }

  SIMPLE_GET_METHOD( Int, Order );
  void setOrder( IntCref aValue );
  
  void setSSystemMatrix( PolymorphCref aValue );
  const Polymorph getSSystemMatrix() const
   {
      return SSystemMatrix;
    }
  
  void initialize()
    {
      Process::initialize();
    }  
  
  void process();
  
 protected:

  Int Order;
  Int theSystemSize;
  mutable bool theChangedFlag;

  Polymorph SSystemMatrix;
  
  // State variables in log space
  gsl_matrix* theY;
  
  // S-System vectors
  gsl_vector* theAlpha;
  gsl_vector* theBeta;
  gsl_matrix* theG;
  gsl_matrix* theH;
  
  // tmp S-System vectors
  gsl_matrix* theAlphaBuffer;
  gsl_matrix* theBetaBuffer;
  gsl_matrix* theGBuffer;
  gsl_matrix* theHBuffer;
  gsl_matrix* theFBuffer;
  
};

ECELL3_DM_INIT;

using namespace libecs;

void ECELL3_DM_CLASSNAME::setOrder( IntCref aValue ) 
{ 
  Order = aValue;

  if( theChangedFlag )
    {
      // free Matrix & Vactor
      gsl_vector_free( theAlpha );
      gsl_vector_free( theBeta );
      gsl_matrix_free( theY );
      gsl_matrix_free( theG );
      gsl_matrix_free( theH );
      gsl_matrix_free( theAlphaBuffer );
      gsl_matrix_free( theBetaBuffer );
      gsl_matrix_free( theGBuffer );
      gsl_matrix_free( theHBuffer );
      gsl_matrix_free( theFBuffer );
    }

  // init Substance Vector
  theY = gsl_matrix_calloc (theSystemSize+1,Order+1);

  // init S-System Vector & Matrix
  theAlpha = gsl_vector_calloc (theSystemSize+1);
  theBeta  = gsl_vector_calloc (theSystemSize+1);
  theG = gsl_matrix_calloc (theSystemSize+1, theSystemSize+1);
  theH = gsl_matrix_calloc (theSystemSize+1, theSystemSize+1);

  // init S-System tmp Vector & Matrix
  theAlphaBuffer = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theBetaBuffer  = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theGBuffer     = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theHBuffer     = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theFBuffer     = gsl_matrix_calloc (Order+1,Order);

  theChangedFlag = true;

}

void ECELL3_DM_CLASSNAME::setSSystemMatrix( PolymorphCref aValue )
{
  SSystemMatrix = aValue;
  PolymorphVector aValueVector( aValue.asPolymorphVector() );
  theSystemSize = aValueVector.size();

  if( theChangedFlag )
    {
      // free Matrix & Vactor
      gsl_vector_free( theAlpha );
      gsl_vector_free( theBeta );
      gsl_matrix_free( theY );
      gsl_matrix_free( theG );
      gsl_matrix_free( theH );
      gsl_matrix_free( theAlphaBuffer );
      gsl_matrix_free( theBetaBuffer );
      gsl_matrix_free( theGBuffer );
      gsl_matrix_free( theHBuffer );
      gsl_matrix_free( theFBuffer );
    }

  // init Substance Vector
  theY = gsl_matrix_calloc (theSystemSize+1,Order+1);

  // init S-System Vector & Matrix
  theAlpha = gsl_vector_calloc (theSystemSize+1);
  theBeta  = gsl_vector_calloc (theSystemSize+1);
  theG = gsl_matrix_calloc (theSystemSize+1, theSystemSize+1);
  theH = gsl_matrix_calloc (theSystemSize+1, theSystemSize+1);

  // init S-System tmp Vector & Matrix
  theAlphaBuffer = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theBetaBuffer  = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theGBuffer     = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theHBuffer     = gsl_matrix_calloc (theSystemSize+1, Order+1);
  theFBuffer     = gsl_matrix_calloc (Order+1,Order);

  theChangedFlag = true;

  // init Factorial matrix
  for(int m( 2 ) ; m < Order+1 ; m++)
  {
    for(int q( 1 ); q < m ; q++)
    {
      const Real aFact( 1 / gsl_sf_fact(q-1) * gsl_sf_fact(m-q-1) * m * (m-1) );
      gsl_matrix_set( theFBuffer,m ,q ,aFact );
     }
  }

  // set Alpha, Beta, G, H 
  for( Int i( 0 ); i < theSystemSize; i++ )
    {
      gsl_vector_set( theAlpha, i+1, (aValueVector[i].asPolymorphVector())[0].asReal() );
      for( Int j( 0 ); j < theSystemSize; j++ )	
	{
	  if( i == j )
	    {
	      gsl_matrix_set( theG, i+1, j+1, (aValueVector[i].asPolymorphVector())[j+1].asReal() - 1 );
	    }else{
	      gsl_matrix_set( theG, i+1, j+1, (aValueVector[i].asPolymorphVector())[j+1].asReal() );
	    }
	}
      gsl_vector_set( theBeta, i+1, (aValueVector[i].asPolymorphVector())[1+theSystemSize].asReal() );
      for( Int j( 0 ); j < theSystemSize; j++ )	
	{
	  if( i == j )
	    {
	      gsl_matrix_set( theH, i+1, j+1, (aValueVector[i].asPolymorphVector())[2+j+theSystemSize].asReal() -1 );
	    }else{
	      gsl_matrix_set( theH, i+1, j+1, (aValueVector[i].asPolymorphVector())[2+j+theSystemSize].asReal() );
	    }
	}
    }
}

void ECELL3_DM_CLASSNAME::process()
{
  //get theY
  Int anIndex( 0 );
  for( VariableReferenceVectorConstIterator
	 i ( thePositiveVariableReferenceIterator );
       i != theVariableReferenceVector.end() ; ++i )
    {
      if( (*i).getValue() <= 0 )
	{
	  THROW_EXCEPTION( ValueError, "Error:in SSystemPProcess::process().log() in 0.");
	}

      gsl_matrix_set(theY, anIndex, 0, 
		     gsl_sf_log( (*i).getValue() ) );
      anIndex++;
    }

  //differentiate first order
  for(int i( 1 ) ; i < theSystemSize+1 ; i++ )
    {
      double aGt( 0.0 );
      double aHt( 0.0 );
      for(int j( 1 ) ; j < theSystemSize+1 ; j++ )
        {
          aGt += gsl_matrix_get(theG, i, j) * gsl_matrix_get(theY, j-1, 0);
          aHt += gsl_matrix_get(theH, i, j) * gsl_matrix_get(theY, j-1, 0);
        }

      double aAlpha = gsl_vector_get(theAlpha,i) * exp(aGt);
      double aBate  = gsl_vector_get(theBeta,i) * exp(aHt);
      gsl_matrix_set(theAlphaBuffer, i, 1, aAlpha);
      gsl_matrix_set(theBetaBuffer, i, 1, aBate);
      gsl_matrix_set(theY, i-1, 1, aAlpha - aBate);
    }

  //differentiate second and/or more order
  for( int m( 2 ) ; m <= Order ; m++)
    {
      for(int i( 1 ) ; i < theSystemSize+1 ; i++ )
	{
	  
	  gsl_matrix_set(theGBuffer, i, m-1, 0);
	  gsl_matrix_set(theHBuffer, i, m-1, 0);
	  
	  for(int j( 1 ) ; j < theSystemSize+1 ; j++ )
	    {
	      const Real aY( gsl_matrix_get(theY, j-1, m-1) );
	      const Real aG( gsl_matrix_get(theGBuffer, i, m-1) );
	      const Real aH( gsl_matrix_get(theHBuffer, i, m-1) );
	      
	      gsl_matrix_set(theGBuffer, i, m-1, 
			     aG + gsl_matrix_get(theG, i, j) * aY );
	      gsl_matrix_set(theHBuffer, i, m-1, 
			     aH + gsl_matrix_get(theH, i, j) * aY );
	    }
	}
      
      for(int i( 1 ) ; i < theSystemSize+1 ; i++ )
	{
	  gsl_matrix_set(theAlphaBuffer, i, m, 0);
	  gsl_matrix_set(theBetaBuffer, i, m, 0);
	  
	  for(int q( 1 );  0 > m-q ; q++)
	    {
	      gsl_matrix_set(theAlphaBuffer, i, m, 
			     gsl_matrix_get(theAlphaBuffer, i, m) + 
			     gsl_matrix_get(theFBuffer, m, q) *
			     gsl_matrix_get(theAlphaBuffer, i, m-q) *
			     gsl_matrix_get(theGBuffer, i, m-q));
	      gsl_matrix_set(theBetaBuffer, i, m, 
			     gsl_matrix_get(theBetaBuffer, i, m) + 
			     gsl_matrix_get(theFBuffer, m, q) *
			     gsl_matrix_get(theBetaBuffer, i, m-q) *
			     gsl_matrix_get(theHBuffer, i, m-q));
	    }
	  
	  gsl_matrix_set(theY, i-1, m, 
			 gsl_matrix_get(theAlphaBuffer, i, m) - 
			 gsl_matrix_get(theBetaBuffer, i, m));
	}
      
    }

  const Real aStepInterval( getSuperSystem()->getStepper()->getStepInterval() );
  
  //integrate
  for( Int i( 1 ); i < theSystemSize+1; i++)
    {

      Real aY( 0.0 );
      for( Int m( 1 ); m <= Order ; m++)
        {
	    aY += gsl_matrix_get(theY, i-1, m) * 
	      gsl_sf_pow_int(aStepInterval,m) / gsl_sf_fact(m);
	}
      gsl_matrix_set(theY, i-1, 0, aY + gsl_matrix_get(theY, i-1, 0));
    }
  
  //set value
  anIndex = 0;
  for( VariableReferenceVectorConstIterator
	 i ( thePositiveVariableReferenceIterator );
       i != theVariableReferenceVector.end() ; ++i )
    {
      (*i).setValue( exp( gsl_matrix_get(theY, anIndex, 0) ) );
      anIndex++;
    }
}

