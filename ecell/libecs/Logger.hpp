//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include "libecs.hpp"


/*

 */

#include "DataPoint.hpp"

template <class T, class V> class DataPoint;
template <class T, class V> class Logger;

typedef double Real;                          // FIXME :temp
typedef const Real& (*const DataFuncCptr)( ); // FIXME:temp 

typedef double Time;      // FIXME : temporary use
typedef double ValueType; // FIXME : temporary use 

typedef const Real& (*const funcptr)();

DECLARE_CLASS( Object );

class Object
{
public:
  Object(funcptr fp)
    :
    fptr(fp)
  {
    ;
  }

  funcptr fptr;
};




#if defined(STLDATAPOINTVECTOR)
#include "StlDataPointVector.hpp"
#endif /* END OF STLDATAPOINTVECTOR */



/**
   
 */

template <class T, class V>
class Logger
{

  DECLARE_CLASS( Logger );

public:
#if defined(STLDATAPOINTVECTOR)
  typedef StlDataPointVector<T,V> DataPointVector;
#endif /* END OF STLDATAPOINTVECTOR */

#if defined(VVECTOR)
  DECLARE_TYPE( VVector, DataPointVector );
#endif /* END OF VVECTOR */ 

  typedef typename DataPointVector::containee_type containee_type;
  typedef typename DataPointVector::const_iterator const_iterator;
  typedef typename DataPointVector::iterator iterator;
  typedef typename DataPointVector::size_type size_type;

  
public:

  /**
     Constructor
  */
  
  Logger( ObjectPtr );
  
  /**
     Copy constructor
  */
  
  Logger( LoggerCref );


  /// Destructor

  ~Logger( void );


  /**

   */

  const DataPointVector& getData( void ) const;

  /**

   */

  const DataPointVector& getData( const T& start, const T& end ) const;

  /**

   */

  const DataPointVector& getData( const T& first, const T& last, 
			       const T& interval ) const;


  /**

   */

  void update( void );
  
  /**

   */
  //FIXME temp
  void update( containee_type& dp )
  {
    appendData(dp);
  }

  /**

   */

  void push( const containee_type& x )
  {
    theDataPointVector.push( x );
  }

  /**

   */

  void push( const T& t, const V& v )
  {
    theDataPointVector.push( t, v );
  }

  /**

   */

  const_iterator binary_search( const_iterator begin, const_iterator end,
				const T& t ) const
  {
    return theDataPointVector.binary_search( begin, end, t );
  }

  /**

   */

  const_iterator binary_search( size_type begin, size_type end, 
				const T& t ) const
  {
    return theDataPointVector.binary_search( begin, end, t);
  }


  /**

   */

  const T& getStartTime( void ) const;

  /**

   */

  const T& getEndTime( void ) const;


  /**

   */

  const T& getMinInterval( void ) const
  {
    return theMinimumInterval;
  }

  /**

   */

  const T& getCurrentInterval( void ) const;

protected:
  
  
  /**

   */

  void appendData( const containee_type& );


private:


  /// Default constructor is hidden
  
  Logger( void );

  /// Assignment operator is hidden
  
  Logger& operator=( const Logger<T,V>& );
  

  /**

   */
  const DataPointVector& getDataPointVector( void ) const
  {
    return theDataPointVector;
  }

  /**

   */
  const DataFuncCptr& getDataFuncCptr( void ) const
  {
    return theDataFuncCptr;
  }


private:
  

  //
  // Protected and Private data members follow
  //


private:

  /// Data members

  DataPointVector theDataPointVector;
  DataFuncCptr    theDataFuncCptr;
  T               theMinimumInterval;


};


#endif /* __LOGGER_HPP */

