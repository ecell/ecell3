#if !defined(__DATAPOINT_HPP)
#define __DATAPOINT_HPP


/*

 */

/**

 */


template <class T, class V>
class DataPoint
{


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

  DataPoint( const DataPoint& );


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

  DataPoint& operator=( const DataPoint& );


  bool operator<(const DataPoint& second)
  {
    if(this->getTime() < second.getTime())
      {
	return true;
      }
    return false;
    
  }


  bool operator>(const DataPoint& second)
  {
    if(this->getTime() > second.getTime())
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
