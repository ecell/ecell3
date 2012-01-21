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

#ifndef __CONVERTTO_HPP
#define __CONVERTTO_HPP

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>

#include "libecs/Defs.hpp"
#include "libecs/Util.hpp"

namespace libecs
{

template< typename ToType, typename FromType >
class ConvertTo
{
public:
    ToType operator()( const FromType& aValue )
    {
        // strategy:
        // (1) if both of ToType and FromType are arithmetic, and
        //         are not the same type, use NumericCaster.
        // (2) otherwise, just try StaticCaster.
        return typename boost::mpl::if_<
            boost::mpl::and_<
                // both are arithmetic, and
                boost::mpl::and_<
                    boost::is_arithmetic< FromType >,
                    boost::is_arithmetic< ToType > >,
                // not the same type.
                boost::mpl::not_<
                    boost::is_same< FromType, ToType> > >,
            NumericCaster<ToType,FromType>,
            StaticCaster<ToType,FromType> >::type()( aValue );
    }
};

// from String
template< typename ToType >
class ConvertTo< ToType, String >
{
public:
    ToType operator()( String const& aValue )
    {
        // strategy:
        // (1) if ToType is arithmetic, use LexicalCaster.
        // (2) otherwise try StaticCaster
        return typename boost::mpl::if_< 
            boost::is_arithmetic< ToType >,
            LexicalCaster< ToType, String >,
            StaticCaster< ToType, String > >::type()( aValue );
    }
};

template< typename ToType, std::size_t _N >
class ConvertTo< ToType, char[_N] >
{
public:
    ToType operator()( char const* aValue )
    {
        // strategy:
        // (1) if ToType is arithmetic, use LexicalCaster.
        // (2) otherwise try StaticCaster
        return typename boost::mpl::if_< 
            boost::is_arithmetic< ToType >,
            LexicalCaster< ToType, const char* >,
            StaticCaster< ToType, const char* > >::type()( aValue );
    }
};


template< typename ToType, std::size_t _N >
class ConvertTo< ToType, const char[_N] >
{
public:
    ToType operator()( char const* aValue )
    {
        // strategy:
        // (1) if ToType is arithmetic, use LexicalCaster.
        // (2) otherwise try StaticCaster
        return typename boost::mpl::if_< 
            boost::is_arithmetic< ToType >,
            LexicalCaster< ToType, const char* >,
            StaticCaster< ToType, const char* >
            >::type()( aValue );
    }
};


template< typename ToType >
class ConvertTo< ToType, char* >
{
public:
    ToType operator()( char const* const& aValue )
    {
        // strategy:
        // (1) if ToType is arithmetic, use LexicalCaster.
        // (2) otherwise try StaticCaster
        return typename boost::mpl::if_< 
            boost::is_arithmetic< ToType >,
            LexicalCaster< ToType, const char* >,
            StaticCaster< ToType, const char* >
            >::type()( aValue );
    }
};


template< typename ToType >
class ConvertTo< ToType, char const* >
{
public:
    ToType operator()( char const* const& aValue )
    {
        // strategy:
        // (1) if ToType is arithmetic, use LexicalCaster.
        // (2) otherwise try StaticCaster
        return typename boost::mpl::if_< 
            boost::is_arithmetic< ToType >,
            LexicalCaster< ToType, const char* >,
            StaticCaster< ToType, const char* >
            >::type()( aValue );
    }
};

// to String

template< typename FromType >
class ConvertTo< String, FromType >
{
public:
    String operator()( const FromType& aValue )
    {
        // strategy:
        // (1) if FromType is arithmetic, use LexicalCaster.
        // (2) otherwise try StaticCaster.
        return typename boost::mpl::if_< 
            boost::is_arithmetic< FromType >,
            LexicalCaster< String, FromType >,
            StaticCaster< String, FromType >
            >::type()( aValue );
    }
};


template<>
class ConvertTo< String, String >
{
public:
    String operator()( const String& aValue )
    {
        return aValue;
    }
};



//
// convertTo template function
//
template< typename ToType, typename FromType >
inline ToType convertTo( const FromType& aValue )
{
    return ConvertTo<ToType,FromType>()( aValue );
}

} // namespace libecs

#endif /* __CONVERTTO_HPP */
