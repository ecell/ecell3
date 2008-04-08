//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

#include "libecs.hpp"
#include "Exceptions.hpp"

/**
   @addtogroup util The Utilities.
   Utilities.

   @ingroup libecs
   @{
 */

/** @file */


namespace libecs {

template< class T >
struct select1st: public std::unary_function< T, typename T::first_type >
{
    typedef T argument_type;
    typedef typename T::first_type result_type;

    const typename T::first_type& operator()(const T& val) const {
        return val.first;
    }

    typename T::first_type& operator()(T& val) const {
        return val.first;
    }
};

template< class T >
struct select2nd: public std::unary_function< T, typename T::second_type >
{
    typedef T argument_type;
    typedef typename T::second_type result_type;

    const typename T::second_type& operator()(const T& val) const {
        return val.second;
    }

    typename T::second_type& operator()(T& val) const {
        return val.second;
    }
};

template< class Tmtr_, typename Tmtr_ret_, class T_>
struct unary_compose_impl
    : public std::unary_function<
        typename T_::argument_type,
        typename Tmtr_::result_type >
{
    unary_compose_impl( const Tmtr_& mtr, const T_& f )
        : mtr_( mtr ), f_( f )
    {
    }

    typename Tmtr_::result_type operator()(
            typename T_::argument_type arg ) const
    {
        return mtr_( f_( arg ) );
    }
private:
    Tmtr_ mtr_;
    T_ f_;
};

template< class Tmtr_, class T_>
struct unary_compose_impl< Tmtr_, void, T_ >
    : public std::unary_function< typename T_::argument_type, void >
{
    unary_compose_impl( const Tmtr_& mtr, const T_& f )
        : mtr_( mtr ), f_( f )
    {
    }

    void operator()( typename T_::argument_type arg ) const
    {
        mtr_( f_( arg ) );
    }
private:
    Tmtr_ mtr_;
    T_ f_;
};

template< class Tmtr_, class T_ >
struct unary_compose
    : public unary_compose_impl< Tmtr_, typename Tmtr_::result_type, T_ >
{
    typedef typename T_::argument_type argument_type;
    typedef typename Tmtr_::result_type result_type;

    unary_compose( const Tmtr_& mtr, const T_& f )
        : unary_compose_impl< Tmtr_, typename Tmtr_::result_type, T_ >( mtr, f)
    {
    }
};

template< class Tmtr_, class T_ >
inline unary_compose< Tmtr_, T_ > compose1( const Tmtr_& mtr, const T_& f )
{
    return unary_compose< Tmtr_, T_ >( mtr, f );
}


template< class Tmtr_, class T1_, class T2_ >
struct two_unaries_by_binary_compose
    : public std::binary_function<
        typename T1_::argument_type,
        typename T2_::argument_type,
        typename Tmtr_::result_type >
{
    typedef typename T1_::argument_type first_argument_type;
    typedef typename T2_::argument_type second_argument_type;
    typedef typename Tmtr_::result_type result_type;

    two_unaries_by_binary_compose(
        const Tmtr_& mtr, const T1_& f1, const T2_& f2 )
        : mtr_( mtr ), f1_( f1 ), f2_( f2 )
    {
    }

    typename Tmtr_::result_type operator()(
            first_argument_type arg1,
            second_argument_type arg2 ) const
    {
        return mtr_( f1_( arg1 ), f2_( arg2 ) );
    }
private:
    Tmtr_ mtr_;
    T1_ f1_;
    T2_ f2_;
};

template< typename Tmtr_, typename T1_, typename T2_ >
two_unaries_by_binary_compose< Tmtr_, T1_, T2_ >
compose_2u_to_b(Tmtr_ mtr, T1_ f1, T2_ f2)
{
    return two_unaries_by_binary_compose<Tmtr_, T1_, T2_>(mtr, f1, f2);
}

template< class T_ >
struct deleter: public std::unary_function< T_, void >
{
    typedef T_ argument_type;
    typedef void result_type;

    void operator()( argument_type p ) const
    {
        delete p;
    }
};

template< class T_ >
struct empty_unary_function: public std::unary_function< T_, T_ >
{
    typedef T_ argument_type;
    typedef void result_type;

    T_ operator()( T_ v ) const
    {
        return v;
    }
};

template<typename T1, typename T2>
struct ConstifyTheOtherIfConst
{
    typedef T2 type;
};

template<typename T1, typename T2>
struct ConstifyTheOtherIfConst<const T1, T2>
{
    typedef const T2 type;
};


template< class T >
struct PtrGreater: public std::binary_function< T, T, bool >
{
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef bool result_type;

    bool operator()( T x, T y ) const
    {
        return *y < *x;
    }
};

template< class T >
struct PtrLess: public std::binary_function< T, T, bool >
{
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef bool result_type;

    bool operator()( T x, T y ) const
    {
        return *y > *x;
    }
};

/**
   Check if aSequence's size() is within [ aMin, aMax ].

   If not, throw an OutOfRange exception.

*/
template <class Sequence>
void checkSequenceSize( const Sequence& aSequence,
                        const typename Sequence::size_type aMin,
                        const typename Sequence::size_type aMax )
{
    const typename Sequence::size_type aSize( aSequence.size() );
    if ( aSize < aMin || aSize > aMax )
    {
        throwSequenceSizeError( aSize, aMin, aMax );
    }
}

/**
   Check if aSequence's size() is at least aMin.

   If not, throw an OutOfRange exception.
*/
template <class Sequence>
void checkSequenceSize( const Sequence& aSequence,
                        const typename Sequence::size_type aMin )
{
    const typename Sequence::size_type aSize( aSequence.size() );
    if ( aSize < aMin )
    {
        throwSequenceSizeError( aSize, aMin );
    }
}

///@internal
LIBECS_API void throwSequenceSizeError( const size_t aSize,
                                        const size_t aMin, const size_t aMax );

///@internal
LIBECS_API void throwSequenceSizeError( const size_t aSize, const size_t aMin );

template< typename T >
inline const T nullValue()
{
    return 0;
}

template<>
inline const Real nullValue()
{
    return 0.0;
}

template<>
inline const String nullValue()
{
    return String();
}

/**
   Retrieves the temporary directory from the system settings.
 */
LIBECS_API const char* getTempDirectory();

/**
   Erase white space characters ( ' ', '\t', and '\n' ) from a string
*/
LIBECS_API void eraseWhiteSpaces( StringRef str );


} // namespace libecs

/** @} */

#endif /* __UTIL_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

