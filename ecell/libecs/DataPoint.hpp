#if !defined(__DATAPOINT_HPP)
#define __DATAPOINT_HPP

#include "libecs.hpp"


/*

 */

/**

 */


// DECLARE_TYPE( DataPoint<Float,Float>, FloatDataPoint );

template <class T, class V>
class DataPoint
{

  DECLARE_CLASS( DataPoint );

public:


  /**
     Initializing constructor
     @param 2 objects which are components of DataPoint
  */

  DataPoint( const T&, const V& );


  /**
     Copy constructor
     @param Object constant reference
   */

  DataPoint( DataPointCref );


  /// Destructor

  ~DataPoint( void )
  {
    ; // do nothing
  }


  /**
     Assignments operator
     @param DataPoint constant reference
     @return DataPoint reference
  */

  DataPointRef operator=( DataPointCref );


  bool operator<( DataPointCref second )
  {
    if( getTime() < second.getTime() )
      {
	return true;
      }
    return false;
  }


  bool operator>(const DataPoint& second)
  {
    if( getTime() > second.getTime() )
      {
	return true;
      }
    return false;
  }


  //
  // Accessors
  //

  /**
     Return the data member, theTime
     @return T constant reference
   */

  const T& getTime( void ) const
  {
    return theTime;
  }


  /**
     Return the data member, theValue
     @return V constant reference
   */

  const V& getValue( void ) const
  {
    return theValue;
  }
  
private:

  /// Default constructor prohibited to public use

  DataPoint( void );

  /**

     @param int object
     @return DataPoint reference
   */

  DataPoint& operator[]( int );


  //
  // Mutators
  //

  /**
     Sets the data member, theTime
     @param T const object
   */

  void setTime( const T& t )
  {
    theTime = t;
  }


  /**
     Sets the data member, theValue
     @param V const object
   */

  void setValue( const V& v )
  {
    theValue = v;
  }


  //
  // Private data members follow
  //

private:

  /// The internal value

  T theTime;
  V theValue;
  
};


#endif /* __DATAPOINT_HPP */
