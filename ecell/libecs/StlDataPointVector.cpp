/*





 */

#if !defined(__DATAPOINT_HPP)
#include "StlDataPointVector.hpp"
#endif

#include <algorithm>
#include <stdio.h> // FIXME : for debugging


template <class T, class V> class DataPoint;
template <class T, class V, class Containee, class Container> 
class StlDataPointVector;

template <class T, class V, class Containee, class Container>
StlDataPointVector<T,V,Containee,Container>::
StlDataPointVector( const StlDataPointVector& datapointvector )
  :
  theContainer( datapointvector.theContainer )
{
  ; // do nothing
}
  


// Destructor

template <class T, class V, class Containee, class Container>
StlDataPointVector<T,V,Containee,Container>::~StlDataPointVector( void )
{
  for( iterator i( begin() ) ; i < end(); i++ )
    {
      delete *i;
    }
}

//

template <class T, class V, class Containee, class Container>
void StlDataPointVector<T,V,Containee,Container>::push(const T& t, const V& val)
{
  theContainer.push_back( new Containee( t, val ) );
}

template <class T, class V, class Containee, class Container>
void StlDataPointVector<T,V,Containee,Container>::push(const Containee& x)
{
  theContainer.push_back( new Containee( x ) );
}

//


template <class T, class V, class Containee, class Container>
StlDataPointVector<T,V,Containee,Container>::const_iterator 
StlDataPointVector<T,V,Containee,Container>::
binary_search(const_iterator first, const_iterator last, const T& val) const
{
  V v;
  DataPoint<T,V> dp(val,v);
  const_iterator itr = lower_bound( first, last, &dp );
  return itr;

}


#if defined(STLDATAPOINTVECTOR_TEST)

#include <iostream>
#include "DataPoint.cpp"

typedef double Float;

int main()
{
  DataPoint<Float,Float> dp1 = DataPoint<Float,Float>(3.14,3.14);
  StlDataPointVector<Float,Float> dpvec = StlDataPointVector<Float,Float>(); 
  dpvec.push(0,0);
  dpvec.push(dp1);
  dpvec.push(3.15,3.0);
  dpvec.push(8.5,3.1);
  dpvec.push(100.45, 1.0);
  StlDataPointVector<Float,Float> dpvec_clone = StlDataPointVector<Float,Float>(dpvec); 
  
  StlDataPointVector<Float,Float>::iterator i;
  for(i=dpvec_clone.begin();i<dpvec_clone.end();i++)
    {
      printf("%p getTime = %f, getValue = %f\n",i,(*i)->getTime(),(*i)->getValue());
    }
  for(i=dpvec.begin();i<dpvec.end();i++)
    {
      printf("%p getTime = %f, getValue = %f\n",i,(*i)->getTime(),(*i)->getValue());
    }
  dpvec.binary_search(dpvec.begin(),dpvec.end(),0.4);

}

#endif /* END OF STLDATAPOINTVECTOR_TEST */






