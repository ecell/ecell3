#include <gsl/gsl_sf.h>
#include <vector>

#include "libecs.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "ESSYNSProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( GMAProcess, ESSYNSProcess )
{

 public:

  LIBECS_DM_OBJECT( GMAProcess, Process )
    {
      INHERIT_PROPERTIES( Process );
   
      PROPERTYSLOT_SET_GET( Int, Order );
      PROPERTYSLOT_SET_GET( Polymorph, GMASystemMatrix );
    }
  
  GMAProcess()
    :
    theSystemSize(0),
    Order(3)
    {
      ; // do nothing.
    }

  virtual ~GMAProcess()
    {
      ;
    }

  SIMPLE_GET_METHOD( Int, Order );
  void setOrder( IntCref aValue );
  
  void setGMASystemMatrix( PolymorphCref aValue );

  const Polymorph getGMASystemMatrix() const
    {
      return GMASystemMatrix;
    }
  
  void fire()
    {
      ;
    }

  const std::vector<RealVector>& getESSYNSMatrix();

  Int getSystemSize()
  {
    return theSystemSize;
  }

  void initialize()
    {
      Process::initialize();
    }  
  
 protected:

  Int Order;
  Int theSystemSize;
  Int theLawSize;

  Polymorph GMASystemMatrix;
  
  // State variables in log space
  std::vector< RealVector > theY;
  
  // GMA-System vectors
  std::vector< RealVector > theAlpha;

  std::vector< std::vector< RealVector > > theG;
  
  // tmp GMA-System vectors
  std::vector< std::vector< RealVector > > theAlphaBuffer;
  std::vector< std::vector< RealVector > > theGBuffer;
  std::vector< RealVector > theFBuffer;
  
};

LIBECS_DM_INIT( GMAProcess, Process );

void GMAProcess::setOrder( IntCref aValue ) 
{ 
  Order = aValue;
  
  //init ESSYNS Matrix
  RealVector tmp;
  tmp.resize(Order+1);
  
  // init Substance Vector
  theY.resize(theSystemSize+1);
  tmp.resize(Order+1);
  for(Int i( 0 ); i < theLawSize; i++)
    {
      theY[i] = tmp;
    }

  // init GMA Vector & Matrix
  theAlpha.resize(theLawSize);
  theG.resize( theLawSize);
  tmp.resize(theLawSize);
  for(Int i( 0 ); i < theLawSize; i++)
    {

      theG[i].resize( theLawSize);
      for(Int j( 0 ); j < theLawSize; j++)
	{
	  theG[i][j] = tmp;
	}
      
        theAlpha[i] = tmp;
   }

  // init S-System tmp Vector & Matrix
  theAlphaBuffer.resize( theLawSize);
  theGBuffer.resize( theLawSize);
  tmp.resize(Order+1);
  for(Int i( 0 ); i < theLawSize; i++)
    {
      theAlphaBuffer[i].resize( theLawSize );
      theGBuffer[i].resize( theLawSize );
      for(Int j( 0 ); j < theLawSize; j++)
	{
	  theAlphaBuffer[i][j] = tmp;
	  theGBuffer[i][j] = tmp;
	}
    }

  theFBuffer.resize( Order + 1);
  tmp.resize(Order);
  for(Int i( 0 ); i < Order + 1; i++)
    {
      theFBuffer[i] = tmp;
    }  

}


void GMAProcess::setGMASystemMatrix( PolymorphCref aValue )
{
  GMASystemMatrix = aValue;
  PolymorphVector aValueVector( aValue.asPolymorphVector() );
  theSystemSize = aValueVector.size();
  theLawSize = theSystemSize + 1;

  // init Substance Vector
  theY.resize(theLawSize);
  RealVector tmp;
  tmp.resize(Order+1);
  for(Int i( 0 ); i < theLawSize; i++)
    {
      theY[i] = tmp;
    }

  // init GMA-System Vector & Matrix
  theAlpha.resize( theLawSize );
  theG.resize( theLawSize);
  tmp.resize(theLawSize);
  for(Int i( 0 ); i < theLawSize; i++)
    {
      theAlpha[i] = tmp;
      //      theG[i] = tmp;
      theG[i].resize( theLawSize );

      for(Int j( 0 ); j < theLawSize; j++)
	{
	  theG[i][j] = tmp;
	}
    }

  // init GMA-System tmp Vector & Matrix
  theAlphaBuffer.resize( theLawSize);
  theGBuffer.resize( theLawSize);
  tmp.resize(Order+1);

  for(Int i( 0 ); i < theLawSize; i++)
    {
      // theAlphaBuffer[i] = tmp;
      // theGBuffer[i] = tmp;
      theAlphaBuffer[i].resize( theLawSize );
      theGBuffer[i].resize( theLawSize );

      for(Int j( 0 ); j < theLawSize; j++)
	{
	  theAlphaBuffer[i][j] = tmp;
	  theGBuffer[i][j] = tmp;
	}
    }

  theFBuffer.resize( Order + 1);
  tmp.resize(Order);
  for(Int i( 0 ); i < Order + 1; i++)
    {
      theFBuffer[i] = tmp;
    }  

  // init Factorial matrix
  for(Int m( 2 ) ; m < Order+1 ; m++)
    {
      for(Int q( 1 ); q < m ; q++)
	{
	  const Real aFact( 1 / gsl_sf_fact(q-1) * gsl_sf_fact(m-q-1) * m * (m-1) );
	  (theFBuffer[m])[q] = aFact;      
	}
    }

  // set Alpha, Beta, G, H 
  for( Int i( 0 ); i < theSystemSize ; i++ )
    {

      for( Int j( 0 ); j < theSystemSize; j++ )	
	{
	  theAlpha[i+1][j+1] = (aValueVector[i].asPolymorphVector())[j].asReal() ;	  
	  //std::cout <<'A'<< theAlpha[i+1][j+1]<<std::endl;
	  for (Int k( 0 ); k < theSystemSize; k++)
	    {
	      if( i == k )
		{
		  
		  (theG[i+1])[j+1][k+1] = ((aValueVector[i].asPolymorphVector())[theSystemSize + j ].asPolymorphVector())[k].asReal() -1;  
		  //std::cout <<"  G"<<theG[i+1][j+1][k+1];
		}else{
		  
		  (theG[i+1])[j+1][k+1] = ((aValueVector[i].asPolymorphVector())[theSystemSize + j ].asPolymorphVector())[k].asReal();
		  //std::cout <<"  G"<<theG[i+1][j+1][k+1];
		}	      
	      // std::cout <<std::endl;
	    }
	  
	}
    }
}

const std::vector<RealVector>& GMAProcess::getESSYNSMatrix()
{
  // get theY
  Int anIndex( 0 );
  
  for( VariableReferenceVectorConstIterator
	 i ( thePositiveVariableReferenceIterator );
       i != theVariableReferenceVector.end(); ++i )
    {
      if( (*i).getVariable()->getValue() <= 0 )
	{
	  THROW_EXCEPTION( ValueError, 
			   "Error:in GMAProcess::getESSYNSMatrix().log() in 0." );
	}

      (theY[anIndex])[0] =
	gsl_sf_log( (*i).getVariable()->getValue() ) ;

      anIndex++;
    }

  // differentiate first order
  Real aGt( 0.0 );  
  Real aAlpha( 0.0 );	 
  for( Int i( 1 ); i < theLawSize; i++ )
    {     
      (theY[i-1])[1] = 0;//reset theY
      for( Int j( 1 ) ; j < theLawSize; j++ )
	{
	  aGt = 0.0;//reset aGt
	  for( Int k( 1 ) ; k < theLawSize ; k++ )
	    {
	      aGt += ( (theG[i])[j][k] * (theY[k-1])[0] );
	    }
	  
	  aAlpha = 0.0;
	  aAlpha = ( theAlpha[i][j] * exp( aGt ) );	 
	  
	  (theAlphaBuffer[i])[j][1] = aAlpha;
	  (theY[i-1])[1] +=  aAlpha;
	}
    }

  // differentiate second and/or more order
  for( Int m( 2 ); m <= Order; m++ ) 
    {
      for( Int i( 1 ) ; i < theLawSize; i++ )
       {  
	 for( Int j( 1 ); j < theLawSize; j++ )
	   {
	     (theGBuffer[i])[j][m] = 0; //reset GBuffer	
	     (theAlphaBuffer[i])[j][m] = 0; //reset ABuffer
	     
	     for( Int k( 1 ); k < theLawSize; k++ )
	       {
		 (theGBuffer[i])[j][m-1] += 
		   ( (theG[i])[j][k] * (theY[k-1])[m-1] ); 
	       }
	     for( Int q( 1 );  q <= m-1; q++)
	       {
		 (theAlphaBuffer[i])[j][m] +=  
		   ( (theFBuffer[m])[q]*
		     (theAlphaBuffer[i])[j][m-q]* 
		     (theGBuffer[i])[j][q] );
	       }
	     
	      (theY[i-1])[m] = (theAlphaBuffer[i])[j][m] / (m-1);
	    }
	} 
    }
  
  return theY;

  /*
  const Real aStepInterval( getSuperSystem()->
			    getStepper()->getStepInterval() );


  //integrate
  Real aY( 0.0 ); 
  for( Int i( 1 ); i < theSystemSize+1; i++ )
    {
      aY = 0.0;//reset aY 
      for( Int m( 1 ); m <= Order; m++ )
        {
	  aY += ((theY[i-1])[m] *
		 gsl_sf_pow_int( aStepInterval, m ) / gsl_sf_fact( m ));
	}
      (theY[i-1])[0] += aY;
    }

  //set value
  anIndex = 0;

  for( VariableReferenceVectorConstIterator
	 i ( thePositiveVariableReferenceIterator );
       i != theVariableReferenceVector.end(); ++i )
    {
      (*i).getVariable()->setValue( exp( (theY[anIndex])[0] ) );
      anIndex++;
    }

  */
}
