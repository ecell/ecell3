//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
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

#ifndef __UTIL_HPP
#define __UTIL_HPP

#include <stdlib.h>
#include <sstream>
#include <functional>
#include <limits>

#include <boost/version.hpp>

#if BOOST_VERSION >= 103200 // for boost-1.32.0 or later.
#   include <boost/numeric/conversion/cast.hpp>
#else // use this instead for boost-1.31 or earlier.
#   include <boost/cast.hpp>
#endif

#include <boost/lexical_cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/range/size.hpp>
#include <boost/range/size_type.hpp>
#include <boost/call_traits.hpp>
#include <boost/type_traits/remove_const.hpp>

#include "libecs/Defs.hpp"
#include "libecs/Exceptions.hpp"

namespace libecs
{
/** 
   A universal to String / from String converter.

   Two usages:
   - stringCast( VALUE )        -- convert VALUE to a string.
   - stringCast<TYPE>( STRING ) -- convert STRING to a TYPE object.

   This is a thin wrapper over boost::lexical_cast.
   This stringCast template function has some specializations for
   common numeric types such as Real and Integer are defined, and
   use of this instead of boost::lexical_cast for those types
   can reduce resulting binary size.
*/
template< typename NEW, typename GIVEN >
NEW stringCast( const GIVEN& aValue )
{
    return boost::lexical_cast<NEW>( aValue );
}


/** @internal */
template< typename GIVEN >
String stringCast( const GIVEN& aValue )
{
    return stringCast<String,GIVEN>( aValue );
}

#define __STRINGCAST_SPECIALIZATION_DECL( NEW, GIVEN )\
template<> LIBECS_API NEW stringCast<NEW,GIVEN>( const GIVEN& )

__STRINGCAST_SPECIALIZATION_DECL( String, Real );
__STRINGCAST_SPECIALIZATION_DECL( String, HighReal );
__STRINGCAST_SPECIALIZATION_DECL( String, Integer );
__STRINGCAST_SPECIALIZATION_DECL( String, UnsignedInteger );
__STRINGCAST_SPECIALIZATION_DECL( Real, String );
__STRINGCAST_SPECIALIZATION_DECL( HighReal, String );
__STRINGCAST_SPECIALIZATION_DECL( Integer, String );
__STRINGCAST_SPECIALIZATION_DECL( UnsignedInteger, String );
// __STRINGCAST_SPECIALIZATION_DECL( String, String );

#undef __STRINGCAST_SPECIALIZATION_DECL


/**
   Erase white space characters ( ' ', '\t', and '\n' ) from a string
*/
void eraseWhiteSpaces( String& str );

template < class T >
struct PtrGreater
{
    bool operator()( T x, T y ) const { return *y < *x; }
};


template < class T >
struct PtrLess
{
    bool operator()( T x, T y ) const { return *y > *x; }
};

template < typename T_ >
struct SelectFirst
{
    typedef T_ argument_type;
    typedef typename T_::first_type result_type;

    typename T_::first_type& operator()( T_& pair ) const
    {
        return pair.first;
    }

    typename T_::first_type const& operator()( T_ const& pair ) const
    {
        return pair.first;
    }
};

template < typename T_ >
struct SelectSecond
{
    typedef T_ argument_type;
    typedef typename T_::second_type result_type;

    typename T_::second_type& operator()( T_& pair ) const
    {
        return pair.second;
    }

    typename T_::second_type const& operator()( T_ const& pair ) const
    {
        return pair.second;
    }
};

template < typename Tderived_, typename Tfun1_, typename Tfun2_,
           typename Tretval_ = typename Tfun1_::result_type >
struct UnaryComposeImpl
{
    typedef typename Tfun2_::argument_type argument_type;
    typedef typename Tfun1_::result_type result_type;

    UnaryComposeImpl( Tfun1_ const& f1, Tfun2_ const& f2 )
        : f1_( f1 ), f2_( f2 ) {}

    result_type operator()( argument_type const& val ) const
    {
        return f1_( f2_( val ) );
    }

    result_type operator()( argument_type const& val )
    {
        return f1_( f2_( val ) );
    }

    result_type operator()( argument_type& val ) const
    {
        return f1_( f2_( val ) );
    }

    result_type operator()( argument_type& val )
    {
        return f1_( f2_( val ) );
    }

private:
    Tfun1_ f1_;
    Tfun2_ f2_;
};

template < typename Tderived_, typename Tfun1_, typename Tfun2_ >
struct UnaryComposeImpl< Tderived_, Tfun1_, Tfun2_, void >
{
    typedef typename Tfun2_::argument_type argument_type;
    typedef void result_type;

    UnaryComposeImpl( Tfun1_ const& f1, Tfun2_ const& f2 )
        : f1_( f1 ), f2_( f2 ) {}

    void operator()( argument_type const& val ) const
    {
        f1_( f2_( val ) );
    }

    void operator()( argument_type const& val )
    {
        f1_( f2_( val ) );
    }

    void operator()( argument_type& val ) const
    {
        f1_( f2_( val ) );
    }

    void operator()( argument_type& val )
    {
        f1_( f2_( val ) );
    }

private:
    Tfun1_ f1_;
    Tfun2_ f2_;
};


template < typename Tfun1_, typename Tfun2_ >
struct UnaryCompose: public UnaryComposeImpl< UnaryCompose< Tfun1_, Tfun2_ >,
                                              Tfun1_, Tfun2_ >
{
public:
    UnaryCompose( Tfun1_ const& f1, Tfun2_ const& f2 )
        : UnaryComposeImpl< UnaryCompose, Tfun1_, Tfun2_ >( f1, f2 ) {}
};

template < typename Tfun1_, typename Tfun2_ >
inline UnaryCompose< Tfun1_, Tfun2_ >
ComposeUnary( Tfun1_ const& f1, Tfun2_ const& f2 )
{
    return UnaryCompose< Tfun1_, Tfun2_ >( f1, f2 );
}

template < typename T_ >
struct DeletePtr
{
    typedef void result_type;
    typedef T_* argument_type;

    void operator()( T_* ptr )
    {
        delete ptr;
    }
};

/** @internal */
LIBECS_API void throwSequenceSizeError( const size_t aSize, 
                                        const size_t aMin, const size_t aMax );

/** @internal */
LIBECS_API void throwSequenceSizeError( const size_t aSize, const size_t aMin );

/**
   Check if aSequence's size() is within [ aMin, aMax ].    

   If not, throw a RangeError exception.
*/
template <class Sequence>
void checkSequenceSize( const Sequence& aSequence, 
                        const typename boost::range_size< Sequence >::type aMin,
                        const typename boost::range_size< Sequence >::type aMax )
{
    const typename Sequence::size_type aSize( boost::size( aSequence ) );
    if( aSize < aMin || aSize > aMax )
    {
        throwSequenceSizeError( aSize, aMin, aMax );
    }
}


/**
   Check if aSequence's size() is at least aMin.

   If not, throw a RangeError exception.
*/

template <class Sequence>
void checkSequenceSize( const Sequence& aSequence, 
                        const typename boost::range_size< Sequence >::type aMin )
{
    const typename Sequence::size_type aSize( boost::size( aSequence ) );
    if( aSize < aMin )
    {
        throwSequenceSizeError( aSize, aMin );
    }
}

/**
   Form a 'for' loop over a STL sequence.

   Use this like:

   FOR_ALL( std::vector<int>, anIntVector )
   {
       int anInt( *i ); // the iterator is 'i'.
       ...
   }

   @arg SEQCLASS the classname of the STL sequence. 
   @arg SEQ the STL sequence.
*/
#define FOR_ALL( SEQCLASS, SEQ )\
for( SEQCLASS ::const_iterator i( (SEQ) .begin() ) ;\
        i != (SEQ) .end() ; ++i )

template< typename T >
inline T nullValue()
{
    return 0;
}


template<>
inline Real nullValue()
{
    return 0.0;
}


template<>
inline String nullValue()
{
    return String();
}


template< class NEW, class GIVEN >
class StaticCaster: public std::unary_function< GIVEN, NEW >
{
public:
    inline NEW operator()( const GIVEN& aValue )
    {
        return static_cast<NEW>( aValue );
    }
};

template< class NEW, class GIVEN >
class DynamicCaster: public std::unary_function< GIVEN, NEW >
{
public:
    NEW operator()( const GIVEN& aPtr )
    {
        NEW aNew( dynamic_cast<NEW>( aPtr ) );
        if( aNew )
        {
            return aNew;
        }
        else
        {
            THROW_EXCEPTION( TypeError, "dynamic cast failed" );
        }
    }
};

template< typename NEW, typename GIVEN >
inline NEW SafeDynamicCast( const GIVEN& aPtr )
{
    return DynamicCaster< NEW, GIVEN >()( aPtr );
}

template< class NEW, class GIVEN >
class LexicalCaster: public std::unary_function< GIVEN, NEW >
{
public:
    NEW operator()( const GIVEN& aValue )
    {
        return stringCast<NEW>( aValue );
    }
};



template< class NEW, class GIVEN >
class NumericCaster: public std::unary_function< GIVEN, NEW >
{
public:
    inline NEW operator()( GIVEN aValue )
    {
        return boost::numeric_cast<NEW>( aValue );
    }
};


template< class OP >
class SecondBinder: public std::unary_function< typename OP::result_type, typename OP::first_argument_type >
{
public:
    typedef typename OP::result_type result_type;
    typedef typename OP::first_argument_type argument_type;

public:
	SecondBinder( OP const& anOp, typename boost::call_traits< typename OP::second_argument_type >::param_type aBoundValue )
        : theOp( anOp ), theBoundValue( aBoundValue )
    {
    }

    result_type operator()( typename boost::remove_const< typename OP::first_argument_type >::type const& aFirstArgValue ) const
    {
        return theOp( aFirstArgValue, theBoundValue );
    }

    result_type operator()( typename boost::remove_const< typename OP::first_argument_type >::type& aFirstArgValue ) const
    {
        return theOp( aFirstArgValue, theBoundValue );
    }

private:
    OP theOp;
    typename OP::second_argument_type theBoundValue;
};

template< class OP >
SecondBinder< OP > BindSecond( OP const& anOp,
                               typename boost::call_traits< typename OP::second_argument_type >::param_type aBoundValue )
{
	return SecondBinder< OP >( anOp, aBoundValue );
}

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

template<typename T>
inline T safe_add(const T& o1, const T& o2)
{
    T r = o1 + o2;
    BOOST_ASSERT(r >= o1 || r >= o2);
    return r;
}

template<typename T1, typename T2>
inline T1 safe_mul(const T1& o1, const T2& o2)
{
    T1 r = o1 * o2;
    BOOST_ASSERT(sizeof(o1) >= sizeof(double) || static_cast<T1>(static_cast<double>(o1) * o2) == r);
    return r;
}

template<typename Td, typename Ts>
inline Td safe_int_cast(const Ts& v)
{
    const Td retval = static_cast<Td>( v );
    BOOST_ASSERT(v == static_cast<Ts>( retval ));
    return retval;
}

} // namespace libecs

#endif /* __UTIL_HPP */
