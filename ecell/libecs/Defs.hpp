//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#ifndef __DEFS_HPP
#define __DEFS_HPP

#include <stdint.h>
#include <float.h>
#include <string>
#include <list>
#include <vector>

#include <boost/call_traits.hpp>

#include "ecell/config.h"




#define DEBUG


// stringifiers.  see preprocessor manual
#define XSTR( S ) STR( S )
#define STR( S ) #S


#define USE_LIBECS using namespace libecs

// cmath

#if defined( HAVE_CMATH )
#include <cmath>
#elif defined( HAVE_MATH )
#include <math>
#else
#error "either math or cmath header is needed."
#endif /* HAVE_CMATH */

namespace libecs
{

  // Some macros those origins from CoreLinux++

  /**
     IGNORE_RETURN is an indicator that the return
     value for a function is ignored.
     i.e   IGNORE_RETURN getSomething( ... );
     Eliminates a lint warning.
  */

#define IGNORE_RETURN (void)

  /**
     Declare a new type and its pointer,
     const pointer, reference, and const reference types. For example
     DECLARE_TYPE( Dword, VeryLongTime );
     @param mydecl The base type
     @param mytype The new type
  */

#define DECLARE_TYPE( mydecl, mytype )  \
typedef mydecl         mytype;         \
typedef mytype *       mytype ## Ptr;  \
typedef const mytype * mytype ## Cptr; \
typedef mytype &       mytype ## Ref;  \
typedef const mytype & mytype ## Cref;

  /**
     Declare class , class pointer , 
     const pointer, class reference 
     and const class reference types for classes. For example
     DECLARE_CLASS( Exception );
     @param tag The class being declared
  */

#define DECLARE_CLASS( tag )            \
   class   tag;                        \
   typedef tag *       tag ## Ptr;     \
   typedef const tag * tag ## Cptr;    \
   typedef tag &       tag ## Ref;     \
   typedef const tag & tag ## Cref;


  // *******************************************
  // Define the void pointer type.
  // *******************************************
   
  typedef void * VoidPtr;

  // *******************************************
  // Define the NULLPTR
  // *******************************************

#define  NULLPTR  0
   
  /**
     STL list template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a list.
     @param name The name you want to give the collection
     @param type The type object the collection manages
  */
#define DECLARE_LIST( type, name )                            \
      DECLARE_TYPE(std::list<type>,name);                       \
      typedef name::iterator name ## Iterator;                  \
      typedef name::iterator& name ## IteratorRef;              \
      typedef name::iterator* name ## IteratorPtr;              \
      typedef name::const_iterator name ## ConstIterator;       \
      typedef name::const_iterator& name ## ConstIteratorRef;   \
      typedef name::const_iterator* name ## ConstIteratorPtr;   \
      typedef name::reverse_iterator name ## Riterator;         \
      typedef name::reverse_iterator& name ## RiteratorRef;     \
      typedef name::reverse_iterator* name ## RiteratorPtr


  /**
     STL vector template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a vector.
     @param name The name you want to give the collection
     @param type The type for the vector
  */
#define DECLARE_VECTOR( type, name )                            \
   DECLARE_TYPE(std::vector<type>,name);                       \
   typedef name::iterator name ## Iterator;                    \
   typedef name::iterator& name ## IteratorRef;                \
   typedef name::iterator* name ## IteratorPtr;                \
   typedef name::const_iterator name ## ConstIterator;         \
   typedef name::const_iterator& name ## ConstIteratorRef;     \
   typedef name::const_iterator* name ## ConstIteratorPtr;     \
   typedef name::reverse_iterator name ## Riterator;           \
   typedef name::reverse_iterator& name ## RiteratorRef;       \
   typedef name::reverse_iterator* name ## RiteratorPtr


  /**
     STL set template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a set.
     @param name The name you want to give the collection
     @param key The object that represents the set key
     @param comp The comparator functor
  */
#define DECLARE_SET(key,comp,name)                                       \
      typedef set<key, comp > name;                                           \
      typedef name *       name ## Ptr;                                       \
      typedef const name * name ## Cptr;                                      \
      typedef name &       name ## Ref;                                       \
      typedef const name & name ## Cref;                                      \
      typedef name::iterator name ## Iterator;                                \
      typedef name::iterator& name ## IteratorRef;                            \
      typedef name::iterator* name ## IteratorPtr;                            \
      typedef name::const_iterator name ## ConstIterator;                     \
      typedef name::const_iterator& name ## ConstIteratorRef;                 \
      typedef name::const_iterator* name ## ConstIteratorPtr;                 \
      typedef name::reverse_iterator name ## Riterator;                       \
      typedef name::reverse_iterator& name ## RiteratorRef;                   \
      typedef name::reverse_iterator* name ## RiteratorPtr
   
  /**
     STL multiset template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a multiset.
     @param name The name you want to give the collection
     @param key The object that represents the mutliset key
     @param comp The comparator functor
  */
#define DECLARE_MULTISET(key,comp,name)                                  \
      typedef multiset<key, comp > name;                                      \
      typedef name *       name ## Ptr;                                       \
      typedef const name * name ## Cptr;                                      \
      typedef name &       name ## Ref;                                       \
      typedef const name & name ## Cref;                                      \
      typedef name::iterator name ## Iterator;                                \
      typedef name::iterator& name ## IteratorRef;                            \
      typedef name::iterator* name ## IteratorPtr;                            \
      typedef name::const_iterator name ## ConstIterator;                     \
      typedef name::const_iterator& name ## ConstIteratorRef;                 \
      typedef name::const_iterator* name ## ConstIteratorPtr;                 \
      typedef name::reverse_iterator name ## Riterator;                       \
      typedef name::reverse_iterator& name ## RiteratorRef;                   \
      typedef name::reverse_iterator* name ## RiteratorPtr


  /**
     STL map template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a map.
     @param name The name you want to give the collection
     @param key The object that represents the map key
     @param value The object that the key is associated to
     @param comp The comparator functor
  */
#define DECLARE_MAP(key,value,comp,name)                             \
      typedef std::map<key,value,comp > name;                      \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef name::iterator name ## Iterator;                     \
      typedef name::iterator& name ## IteratorRef;                 \
      typedef name::iterator* name ## IteratorPtr;                 \
      typedef name::const_iterator name ## ConstIterator;          \
      typedef name::const_iterator& name ## ConstIteratorRef;      \
      typedef name::const_iterator* name ## ConstIteratorPtr;      \
      typedef name::reverse_iterator name ## Riterator;            \
      typedef name::reverse_iterator& name ## RiteratorRef;        \
      typedef name::reverse_iterator* name ## RiteratorPtr
   
  /**
     STL multimap template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a multimap.
     @param name The name you want to give the collection
     @param key The object that represents the map key
     @param value The object that the key is associated to
     @param comp The comparator functor
  */

#define DECLARE_MULTIMAP(key,value,comp,name)                 \
      typedef std::multimap<key,value,comp > name;                 \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef name::iterator name ## Iterator;                     \
      typedef name::iterator& name ## IteratorRef;                 \
      typedef name::iterator* name ## IteratorPtr;                 \
      typedef name::const_iterator name ## ConstIterator;          \
      typedef name::const_iterator& name ## ConstIteratorRef;      \
      typedef name::const_iterator* name ## ConstIteratorPtr;      \
      typedef name::reverse_iterator name ## Riterator;            \
      typedef name::reverse_iterator& name ## RiteratorRef;        \
      typedef name::reverse_iterator* name ## RiteratorPtr


  /**
     STL queue template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a queue.
     @param name The name you want to give the collection
     @param type The type to be queued
  */
#define DECLARE_QUEUE( type, name )                          \
      DECLARE_TYPE(std::deque<type>,name);                     \
      typedef name::iterator name ## Iterator;                 \
      typedef name::iterator& name ## IteratorRef;             \
      typedef name::iterator* name ## IteratorPtr;             \
      typedef name::const_iterator name ## ConstIterator;      \
      typedef name::const_iterator& name ## ConstIteratorRef;  \
      typedef name::const_iterator* name ## ConstIteratorPtr;  \
      typedef name::reverse_iterator name ## Riterator;        \
      typedef name::reverse_iterator& name ## RiteratorRef;    \
      typedef name::reverse_iterator* name ## RiteratorPtr

  /**
     STL stack template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a stack.
     @param name The name you want to give the collection
     @param type The type to be stacked
  */
#define DECLARE_STACK( type, name )                                 \
      DECLARE_TYPE(stack<type>,name)                                   


  // from Loki

  
#define DECLARE_ASSOCVECTOR(key,value,comp,name)                             \
      typedef ::Loki::AssocVector<key,value,comp > name;                      \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef name::iterator name ## Iterator;                     \
      typedef name::iterator& name ## IteratorRef;                 \
      typedef name::iterator* name ## IteratorPtr;                 \
      typedef name::const_iterator name ## ConstIterator;          \
      typedef name::const_iterator& name ## ConstIteratorRef;      \
      typedef name::const_iterator* name ## ConstIteratorPtr;      \
      typedef name::reverse_iterator name ## Riterator;            \
      typedef name::reverse_iterator& name ## RiteratorRef;        \
      typedef name::reverse_iterator* name ## RiteratorPtr

#define DECLARE_ASSOCVECTOR_TEMPLATE(key,value,comp,name)                \
      typedef ::Loki::AssocVector<key,value,comp > name;                      \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef typename name::iterator name ## Iterator;                     \
      typedef typename name::iterator& name ## IteratorRef;                 \
      typedef typename name::iterator* name ## IteratorPtr;                 \
      typedef typename name::const_iterator name ## ConstIterator;          \
      typedef typename name::const_iterator& name ## ConstIteratorRef;      \
      typedef typename name::const_iterator* name ## ConstIteratorPtr;      \
      typedef typename name::reverse_iterator name ## Riterator;            \
      typedef typename name::reverse_iterator& name ## RiteratorRef;        \
      typedef typename name::reverse_iterator* name ## RiteratorPtr



  // Types

  template <typename T>
  class Param
  {
  public:
    typedef typename boost::call_traits<T>::param_type type;
  };

  // String

  DECLARE_TYPE( std::string, String );

  DECLARE_TYPE( const char* const, StringLiteral );

  // Numeric types

  DECLARE_TYPE( long int, Integer );
  DECLARE_TYPE( unsigned long int, UnsignedInteger );
  typedef Param<Integer>::type IntegerParam;
  typedef Param<UnsignedInteger>::type UnsignedIntegerParam;


  // these can cause problem when used as template parameters
  //  DECLARE_TYPE( int64_t, Integer );
  //  DECLARE_TYPE( uint64_t, UnsignedInteger );

  //  DECLARE_TYPE( double, Real );
  DECLARE_TYPE( double, Real );
  typedef Param<Real>::type RealParam;

#if defined( HAVE_LONG_DOUBLE )
  DECLARE_TYPE( long double, HighReal );
#else
  DECLARE_TYPE( double, HighReal );
#endif /* defined( HAVE_LONG_DOUBLE ) */
  typedef Param<HighReal>::type HighRealParam;
    
  //  DECLARE_TYPE( HighReal, Time );
  DECLARE_TYPE( Real, Time );
    
  //! Infinity.  Currently this is defined as INFINITY symbol of C99 standard.
  const Real INF( INFINITY );


  //! Avogadro number. 
  const Real N_A( 6.0221367e+23 );

  //! 1 / Avogadro number (reciprocal of N_A)
  const Real N_A_R( 1.0 / N_A );

  // functions

#if defined( FP_FAST_FMA )
  inline const Real FMA( const Real a, const Real b, const Real c )
  {
    return ::fma( a, b, c );
  }
#else
  inline const Real FMA( const Real a, const Real b, const Real c )
  {
    return a * b + c;
  }
#endif /* defined( FP_FAST_FMA ) */


  // MACROS

#if 0

#if !defined( HAVE_PRETTY_FUNCTION )
#define __PRETTY_FUNCTION__ ""
#endif

#endif // 0

  /**
     Converts each type into a unique, insipid type.
     Invocation Type2Type<T> where T is a type.
     Defines the type OriginalType which maps back to T.
     
     taken from loki library.

     @ingroup util
  */

  template <typename T>
  struct Type2Type
  {
    typedef T OriginalType;
  };


} // namespace libecs


#endif /* __DEFS_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



