#define GSL_RANGE_CHECK_OFF

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <map>

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "QuasiDynamicFluxProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FluxDistributionProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( FluxDistributionProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Polymorph, KnownProcessList );
      PROPERTYSLOT_SET_GET( Polymorph, UnknownProcessList );
      PROPERTYSLOT_SET_GET( Real, Epsilon );
    }

  FluxDistributionProcess();
  ~FluxDistributionProcess();

  SET_METHOD( Polymorph, KnownProcessList )
  {
    KnownProcessList.clear();
    for( Integer i(0); i < value.asPolymorphVector().size(); ++i)
      {
	KnownProcessList.push_back( value.asPolymorphVector()[i].asString() );
      }
  }
  
  GET_METHOD( Polymorph, KnownProcessList )
  {
    PolymorphVector value;
    for( Integer i(0); i < KnownProcessList.size(); ++i)
      {
	value.push_back( KnownProcessList[i] );
      }
    return  value;
  }  
  SET_METHOD( Polymorph, UnknownProcessList )
  {
    UnknownProcessList.clear();
    for( Integer i(0); i < value.asPolymorphVector().size(); ++i)
      {
	UnknownProcessList.push_back( value.asPolymorphVector()[i].asString() );
      }
  }
  
  GET_METHOD( Polymorph, UnknownProcessList )
  {
    PolymorphVector value;
    for( Integer i(0); i < UnknownProcessList.size(); ++i)
      {
	value.push_back( UnknownProcessList[i] );
      }
    return  value;
  }  

  SIMPLE_SET_GET_METHOD( Real, Epsilon );

  void initialize();  
  void fire();
  
 protected:

  gsl_matrix* generateInverse( gsl_matrix *m_unknown, 
			       Integer matrix_size );

 protected:

  StringVector KnownProcessList;
  StringVector UnknownProcessList;

  gsl_matrix* theKnownMatrix;
  gsl_matrix* theUnknownMatrix;
  gsl_matrix* theInverseMatrix;
  gsl_matrix* theSolutionMatrix;

  gsl_matrix* theTmpInverseMatrix;
  gsl_matrix* theTmpUnknownMatrix;

  gsl_vector* theKnownVelocityVector;
  gsl_vector* theSolutionVector;
  gsl_vector* theFluxVector;

  ProcessVector theKnownProcessPtrVector;
  std::vector< QuasiDynamicFluxProcessPtr > theUnknownProcessPtrVector;

  bool   theIrreversibleFlag;
  Integer    theMatrixSize;
  Real   Epsilon;

};

LIBECS_DM_INIT( FluxDistributionProcess, Process );

using namespace libecs;

FluxDistributionProcess::FluxDistributionProcess()
  :
  theMatrixSize( 0 ),
  theIrreversibleFlag( false ),
  theKnownMatrix( NULLPTR ),
  theUnknownMatrix( NULLPTR ),
  theInverseMatrix( NULLPTR ),
  theSolutionMatrix( NULLPTR ),
  theTmpInverseMatrix( NULLPTR ),
  theTmpUnknownMatrix( NULLPTR ),
  theKnownVelocityVector( NULLPTR ),
  theSolutionVector( NULLPTR ),
  theFluxVector( NULLPTR ),
  Epsilon( 1e-6 )
{
  ; // do nothing
}

FluxDistributionProcess::~FluxDistributionProcess()
{
  gsl_matrix_free( theUnknownMatrix );
  gsl_matrix_free( theKnownMatrix );
  gsl_matrix_free( theInverseMatrix );
  gsl_matrix_free( theSolutionMatrix );
  gsl_matrix_free( theTmpInverseMatrix );
  gsl_matrix_free( theTmpUnknownMatrix );
  gsl_vector_free( theKnownVelocityVector );
  gsl_vector_free( theSolutionVector );
  gsl_vector_free( theFluxVector );
}

gsl_matrix* FluxDistributionProcess::generateInverse(gsl_matrix *m_unknown, 
						     Integer matrix_size)
{      
  gsl_matrix *m_tmp1, *m_tmp2, *m_tmp3;
  gsl_matrix *inv;
  gsl_matrix *V; 
  gsl_vector *S, *work;
  
  m_tmp1 = gsl_matrix_calloc(matrix_size,matrix_size);
  m_tmp2 = gsl_matrix_calloc(matrix_size,matrix_size);
  m_tmp3 = gsl_matrix_calloc(matrix_size,matrix_size);
  inv = gsl_matrix_calloc(matrix_size,matrix_size);
  
  V = gsl_matrix_calloc (matrix_size,matrix_size); 
  S = gsl_vector_calloc (matrix_size);             
  work = gsl_vector_calloc (matrix_size);          
      
  gsl_matrix_memcpy(m_tmp1,m_unknown);
  gsl_linalg_SV_decomp(m_tmp1,V,S,work);
      
  //generate Singular Value Matrix
  
  for(int i=0;i<matrix_size;i++){
    double singular_value = gsl_vector_get(S,i);
    if(singular_value > Epsilon){
      gsl_matrix_set(m_tmp2,i,i,1/singular_value);
    }
    else
      {
	gsl_matrix_set(m_tmp2,i,i,0.0);
      }
  }
  
  gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,m_tmp2,m_tmp1,0.0,m_tmp3);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,V,m_tmp3,0.0,inv);
  
  gsl_matrix_free(m_tmp1);
  gsl_matrix_free(m_tmp2);
  gsl_matrix_free(m_tmp3);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work);
  
  return inv;  
}

void FluxDistributionProcess::initialize()
{
  
  Process::initialize();
  
  //
  // generate Variable Map
  //

  std::map< VariablePtr, Integer > aVariableMap;
  std::vector< VariableReferenceVector >  aVariableVector;

  for(Integer i(0); i < UnknownProcessList.size(); i++ ) 
    {
      const FullID aFullID( UnknownProcessList[i] );
      
      SystemPtr aSystem( getSuperSystem()->
			 getSystem( aFullID.getSystemPath() ) );

      ProcessPtr aProcess( aSystem->getProcess( aFullID.getID() ) );
      VariableReferenceVector aVector( aProcess->getVariableReferenceVector());  
      aVariableVector.push_back( aVector );
    }
  
  for( Integer i( 0 ); i < KnownProcessList.size(); i++ ) 
    {
      const FullID aFullID( KnownProcessList[i] );
      SystemPtr aSystem( getSuperSystem()->
			 getSystem( aFullID.getSystemPath() ) );
      
      ProcessPtr aProcess( aSystem->getProcess( aFullID.getID() ) );
      theKnownProcessPtrVector.push_back( aProcess );
      VariableReferenceVector aVector( aProcess->getVariableReferenceVector() );
      aVariableVector.push_back( aVector );
    }
  
  VariableReferenceVector aVariableReferenceVector;
  for( Integer i( 0 ); i < aVariableVector.size(); ++i )
    {
      for( Integer j( 0 ); j < aVariableVector[i].size(); ++j )
	{
	  Integer aFlag( 1 );
	  for( Integer k( 0 ); k < aVariableReferenceVector.size(); ++k )
	    {
	      if( aVariableVector[i][j].getVariable() == aVariableReferenceVector[k].getVariable() )
		{
		  aFlag = 0;
		  break;
		}
	    }
           
          if( aFlag )
            {
              aVariableReferenceVector.push_back( aVariableVector[i][j] );
            }
        }
    }
  
  for(Integer i( 0 ); i < aVariableReferenceVector.size(); i++ )
    {
      aVariableMap.insert( std::pair< VariablePtr, Integer >( aVariableReferenceVector[i].getVariable(), i  ) );
    }
  
  //
  // gsl Matrix & Vector allocation
  //

  if( UnknownProcessList.size() > aVariableReferenceVector.size() )
    {
      theMatrixSize = UnknownProcessList.size();
    }else{
      theMatrixSize = aVariableReferenceVector.size(); 
    }

  if( theUnknownMatrix != NULLPTR )
    { 
      gsl_matrix_free( theUnknownMatrix );
      gsl_matrix_free( theKnownMatrix );
      gsl_matrix_free( theTmpInverseMatrix );
      gsl_matrix_free( theSolutionMatrix );
      gsl_matrix_free( theTmpUnknownMatrix );
      
      gsl_vector_free( theKnownVelocityVector );
      gsl_vector_free( theSolutionVector );
      gsl_vector_free( theFluxVector );
    }
  
  theUnknownMatrix = gsl_matrix_calloc( theMatrixSize, theMatrixSize );
  theKnownMatrix = gsl_matrix_calloc( theMatrixSize, KnownProcessList.size() );

  if( KnownProcessList.size() < theMatrixSize )
    {
      theSolutionMatrix = gsl_matrix_calloc( theMatrixSize, KnownProcessList.size() );    
    }
  else
    {
      theSolutionMatrix = gsl_matrix_calloc( theMatrixSize, theMatrixSize );
    }

  theTmpInverseMatrix = gsl_matrix_calloc( theMatrixSize, theMatrixSize );
  theTmpUnknownMatrix = gsl_matrix_calloc( theMatrixSize, theMatrixSize );
  theKnownVelocityVector = gsl_vector_calloc( KnownProcessList.size() );
  theSolutionVector = gsl_vector_calloc( theMatrixSize );
  theFluxVector = gsl_vector_calloc( theMatrixSize );
  
  //
  // generate theUnknownMatrix
  //

  theUnknownProcessPtrVector.clear();
  
  for(Integer i(0); i < UnknownProcessList.size(); i++ ) 
    {
      const FullID aFullID( UnknownProcessList[i] );
      
      SystemPtr aSystem( getSuperSystem()->
			 getSystem( aFullID.getSystemPath() ) );

      QuasiDynamicFluxProcessPtr aProcess( dynamic_cast<QuasiDynamicFluxProcessPtr>(aSystem->getProcess( aFullID.getID() ) ) );

      theUnknownProcessPtrVector.push_back( aProcess );
      VariableReferenceVector aVector( aProcess->getVariableReferenceVector());
      for( Integer j(0); j < aVector.size(); j++ )
	{
	  gsl_matrix_set( theUnknownMatrix, aVariableMap.find( aVector[j].getVariable() )->second, i, aVector[j].getCoefficient() );
	}
    }
  
  //
  // check Irreversible
  //

  theIrreversibleFlag = false;
  for( Integer i( 0 ); i < UnknownProcessList.size(); i++ )
    {
      if( theUnknownProcessPtrVector[i]->getIrreversible() )
	{
	  theIrreversibleFlag = true;
	}
    }
  
  //
  // generate KnownProcess Matrix
  //

  theKnownProcessPtrVector.clear();
  
  for( Integer i( 0 ); i < KnownProcessList.size(); i++ ) 
    {
      const FullID aFullID( KnownProcessList[i] );
      SystemPtr aSystem( getSuperSystem()->
			 getSystem( aFullID.getSystemPath() ) );
      
      ProcessPtr aProcess( aSystem->getProcess( aFullID.getID() ) );
      theKnownProcessPtrVector.push_back( aProcess );
      VariableReferenceVector aVector( aProcess->getVariableReferenceVector() );

      for( Integer j(0); j < aVector.size(); j++ )
	{
	  gsl_matrix_set( theKnownMatrix, aVariableMap.find( aVector[j].getVariable() )->second, i, aVector[j].getCoefficient());
	}
    }
  
  //
  // generate inverse matrix
  //

  if( theInverseMatrix != NULLPTR )
    { 
      gsl_matrix_free( theInverseMatrix );
    }

  theInverseMatrix = generateInverse( theUnknownMatrix , theMatrixSize );
  
  gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, -1.0, theInverseMatrix, 
		  theKnownMatrix, 0.0, theSolutionMatrix );

}  

void FluxDistributionProcess::fire()
{
  for( Integer i( 0 ); i < theKnownProcessPtrVector.size(); ++i )
    {
      gsl_vector_set( theKnownVelocityVector, i, theKnownProcessPtrVector[i]->getActivity() );
    }
  
  gsl_blas_dgemv( CblasNoTrans, 1.0, theSolutionMatrix, 
		  theKnownVelocityVector, 0.0, theSolutionVector );

  if ( theIrreversibleFlag )
    {
      bool aFlag( false );
      std::vector<Integer> aVector;
      for( Integer i( 0 ); i < theUnknownProcessPtrVector.size(); ++i )
	{
	  if( ( theUnknownProcessPtrVector[i]->getIrreversible() ) &&
	      ( gsl_vector_get(theSolutionVector, i) < 0 ) )
	    {
	      aFlag = true;
	      aVector.push_back(i);
	    }
	}		    
      
      if( aFlag )
	{
	  
	  if( theTmpUnknownMatrix != NULLPTR )
	    { 
	      gsl_matrix_free( theTmpUnknownMatrix );
	    }
	  
	  gsl_matrix_memcpy( theTmpUnknownMatrix, theUnknownMatrix );
	  for( Integer i( 0 ); i < aVector.size(); ++i )
	    {
	      for( Integer j( 0 ); j < theMatrixSize; ++j )
		{
		  gsl_matrix_set( theTmpUnknownMatrix, j, aVector[i], 0 );
		}
	    }
	  
	  if( theTmpInverseMatrix != NULLPTR )
	    { 
	      gsl_matrix_free( theTmpInverseMatrix );
	    }
	  
	  theTmpInverseMatrix = generateInverse( theTmpUnknownMatrix,
						 theMatrixSize );    
	  gsl_blas_dgemm( CblasNoTrans, CblasNoTrans, -1.0, 
			  theTmpInverseMatrix, 
			  theKnownMatrix, 0.0, theSolutionMatrix );
	  gsl_blas_dgemv( CblasNoTrans, 1.0, theSolutionMatrix, 
			  theKnownVelocityVector, 0.0, theSolutionVector );
	}	
    }

  for( Integer i( 0 ); i < theUnknownProcessPtrVector.size(); ++i )
    {
      theUnknownProcessPtrVector[i]->setFlux( gsl_vector_get( theSolutionVector, i ) ) ;
    }
  
}
