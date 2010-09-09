//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
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

#include "libecs/Defs.hpp"

namespace libecs {

template < class CLASS, typename RET, typename ARG1 = void > 
class MethodProxy;

template < class CLASS, typename RET >
class MethodProxy<CLASS, RET>
{
private:
    typedef RET (* Invoker )( CLASS* );

public:
    RET operator()( CLASS* anObject ) const 
    { 
        return theInvoker( anObject ); 
    }

    RET operator()( CLASS const* anObject ) const 
    { 
        return theInvoker( const_cast<CLASS*>( anObject ) ); 
    }

    template< RET (CLASS::*METHOD)() >
    static MethodProxy create()
    {
        return MethodProxy( &invoke<METHOD> );
    }

    template< RET (CLASS::*METHOD)() const >
    static MethodProxy createConst()
    {
        return MethodProxy( reinterpret_cast< Invoker >( &invokeConst<METHOD> ) );
    }

    inline bool operator==( MethodProxy const& that ) const
    {
        return that.theInvoker == theInvoker;
    }

private:
    MethodProxy()
        : theInvoker( 0 )
    {
        ; // do nothing
    }
        
    MethodProxy( Invoker anInvoker ) 
        : theInvoker( anInvoker )
    {
        ; // do nothing
    }

    template< RET (CLASS::*METHOD)() >
    inline static RET invoke( CLASS* anObject )
    {
        return ( anObject->*METHOD )();
    }

    template< RET (CLASS::*METHOD)() const >
    inline static RET invokeConst( CLASS const* anObject )
    {
        return ( anObject->*METHOD )();
    }

private:
    Invoker theInvoker;
};


template < typename RET, typename ARG1 = void > 
class ObjectMethodProxy;

template < typename RET >
class ObjectMethodProxy<RET>
{
private:
    typedef RET (* Invoker )( void* );

public:
    RET operator()() const 
    { 
        return theInvoker( theObject ); 
    }

	template < typename T, RET (T::*TMethod)() >
    static ObjectMethodProxy create( T* anObject )
    {
        return ObjectMethodProxy( reinterpret_cast< Invoker >( &invoke< T, TMethod > ), anObject );
    }

	template < typename T, RET (T::*TMethod)() const >
    static ObjectMethodProxy createConst( T const* anObject )
    {
        return ObjectMethodProxy( reinterpret_cast< Invoker >( &invokeConst< T, TMethod > ), const_cast< T* >( anObject ) );
    }

	inline bool operator==( ObjectMethodProxy const& that ) const
    {
        return that.theInvoker == theInvoker;
    }

private:
    ObjectMethodProxy()
        : theInvoker( 0 ),
          theObject( 0 )
    {
        ; // do nothing
    }
        
    ObjectMethodProxy( Invoker anInvoker, void* anObject ) 
        : theInvoker( anInvoker ),
          theObject( anObject )
    {
        ; // do nothing
    }

    template < class T, RET (T::*TMethod)() >
    static RET invoke( T* anObject )
    {
        return ( anObject->*TMethod )();
    }

    template < class T, RET (T::*TMethod)() const >
    static RET invokeConst( T const* anObject )
    {
        return ( anObject->*TMethod )();
    }

private:
    Invoker     theInvoker;
    void*       theObject;
};

} // namespace libecs

#endif /* __METHODPROXY_HPP */
