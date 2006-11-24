//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2004 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
//END_HEADER
// 
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __METHODPROXY_HPP
#define __METHODPROXY_HPP

#include "libecs.hpp"


template < class CLASS, typename RET, typename ARG1 = void > 
class MethodProxy;

template < class CLASS, typename RET >
class MethodProxy<CLASS,RET>
{
private:

  typedef const RET (* Invoker )( CLASS* const );

public:

  inline const RET operator()( CLASS* const anObject ) const 
  { 
    return theInvoker( anObject ); 
  }

  inline const RET operator()( const CLASS* const anObject ) const 
  { 
    return theInvoker( const_cast<CLASS* const>( anObject ) ); 
  }

  template < const RET (CLASS::*METHOD)() const >
  static MethodProxy create()
  {
#if defined( LIBECS_USE_PMF_CONVERSIONS )
    return MethodProxy( Invoker( METHOD ));
#else  /* defined( LIBECS_USE_PMF_CONVERSIONS ) */
    return MethodProxy( invoke<METHOD> );
#endif /* defined( LIBECS_USE_PMF_CONVERSIONS ) */
  }

private:

  MethodProxy()
    : 
    theInvoker( 0 )
  {
    ; // do nothing
  }
    
  MethodProxy( Invoker anInvoker ) 
    : 
    theInvoker( anInvoker )
  {
    ; // do nothing
  }

  template < const RET (CLASS::*METHOD)() const >
  inline static const RET invoke( CLASS* const anObject )
  {
    return ( anObject->*METHOD )();
  }

private:
    
  Invoker   theInvoker;

};



template < typename RET, typename ARG1 = void > 
class ObjectMethodProxy;

template < typename RET >
class ObjectMethodProxy<RET>
{
private:

  typedef const RET (* Invoker )( void* const );

public:

  inline const RET operator()() const 
  { 
    return theInvoker( theObject ); 
  }

  template < class T, const RET (T::*TMethod)() const >
  static ObjectMethodProxy create( T* const anObject )
  {
#if defined( LIBECS_USE_PMF_CONVERSIONS )
    return ObjectMethodProxy( Invoker( TMethod ), anObject );
#else  /* defined( LIBECS_USE_PMF_CONVERSIONS ) */
    return ObjectMethodProxy( invoke<T,TMethod>, anObject );
#endif /* defined( LIBECS_USE_PMF_CONVERSIONS ) */
  }

private:

  ObjectMethodProxy()
    : 
    theInvoker( 0 ),
    theObject( 0 )
  {
    ; // do nothing
  }
    
  ObjectMethodProxy( Invoker anInvoker, void* anObject ) 
    : 
    theInvoker( anInvoker ),
    theObject( anObject )
  {
    ; // do nothing
  }

  template < class T, const RET (T::*TMethod)() const >
  inline static const RET invoke( void* const anObject )
  {
    return ( static_cast<T*>(anObject)->*TMethod )();
  }

private:
    
  Invoker   theInvoker;
  void*     theObject;

};




#endif /* __METHODPROXY_HPP */
