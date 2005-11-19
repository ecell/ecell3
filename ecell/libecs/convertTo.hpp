//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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

#ifndef __CONVERTTO_HPP
#define __CONVERTTO_HPP

#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include "libecs.hpp"
#include "Util.hpp"

namespace libecs
{


  /** @addtogroup property
      
  @ingroup libecs
  @{
  */



  template< typename ToType, typename FromType >
  class ConvertTo
  {
  public:
    inline const ToType operator()( const FromType& aValue )
    {
      // strategy:
      // (1) if both of ToType and FromType are arithmetic, and
      //     are not the same type, use NumericCaster.
      // (2) otherwise, just try StaticCaster.

      typedef typename boost::mpl::if_c<
	( boost::is_arithmetic<FromType>::value &&
	  boost::is_arithmetic<ToType>::value ) &&  // both are arithmetic, and
	! boost::is_same<FromType,ToType>::value,   // not the same type.
	NumericCaster<ToType,FromType>,
	StaticCaster<ToType,FromType>
	>::type
	Converter;
      
      return Converter()( aValue );
    }
  };


  // ConvertTo specializations


  // for String

  // from String

  template< typename ToType >
  class ConvertTo< ToType, String >
  {
  public:
    inline const ToType operator()( StringCref aValue )
    {
      // strategy:
      // (1) if ToType is arithmetic, use LexicalCaster.
      // (2) otherwise try StaticCaster

      typedef typename boost::mpl::if_< 
	boost::is_arithmetic< ToType >,
	LexicalCaster< ToType, String >,     // is arithmetic
	StaticCaster< ToType, String >       // is not
	>::type
	Converter;

      return Converter()( aValue );
    }
  };
  
  // to String

  template< typename FromType >
  class ConvertTo< String, FromType >
  {
  public:
    inline const String operator()( const FromType& aValue )
    {
      // strategy:
      // (1) if FromType is arithmetic, use LexicalCaster.
      // (2) otherwise try StaticCaster.

      typedef typename boost::mpl::if_< 
	boost::is_arithmetic< FromType >,
	LexicalCaster< String, FromType >,
	StaticCaster< String, FromType >
	>::type
	Converter;

      return Converter()( aValue );
    }
  };


  template<>
  class ConvertTo< String, String >
  {
  public:
    inline const String operator()( const String& aValue )
    {
      return aValue;
    }
  };



  //
  // convertTo template function
  //
  template< typename ToType, typename FromType >
  inline const ToType convertTo( const FromType& aValue )
  {
    return ConvertTo<ToType,FromType>()( aValue );
  }



}


#endif /* __CONVERTTO_HPP */
