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

#ifndef __PRIMITIVETYPE_HPP 
#define __PRIMITIVETYPE_HPP 

#include "libecs.hpp"

namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  class PrimitiveType
  {

  public:
    
    enum Type
      {
	ENTITY    = 1,
	SUBSTANCE = 2,
	REACTOR   = 3,
	SYSTEM    = 4
      };

    PrimitiveType( StringCref typestring );

    PrimitiveType( const Int number );

    PrimitiveType( const Type type )
      :
      theType( type )
    {
      ; // do nothing
    }

    PrimitiveType( PrimitiveTypeCref primitivetype )
      :
      theType( primitivetype.getType() )
    {
      ; // do nothing
    }

    PrimitiveType()
      :
      theType( ENTITY )
    {
    }

      
    StringCref getString() const;

    operator StringCref() const
    {
      return getString();
    }

    const Type& getType() const
    {
      return theType;
    }

    operator const Type&() const
    {
      return getType();
    }

    //    operator const Int&() const
    //    {
    //      return static_cast<const Int&>( getType() ); 
    //    }

    bool operator<( PrimitiveTypeCref rhs ) const
    {
      return theType < rhs.getType();
    }

    bool operator<( const Type rhs ) const
    {
      return theType < rhs;
    }

    bool operator==( PrimitiveTypeCref rhs ) const
    {
      return theType == rhs.getType();
    }

    bool operator==( const Type rhs ) const
    {
      return theType == rhs;
    }


    inline static StringCref  PrimitiveTypeStringOfEntity()
    {
      const static String aString( "Entity" );
      return aString;
    }

    inline static StringCref  PrimitiveTypeStringOfReactor()
    {
      const static String aString( "Reactor" );
      return aString;
    }
    
    inline static StringCref  PrimitiveTypeStringOfSubstance()
    {
      const static String aString( "Substance" );
      return aString;
    }
    
    inline static StringCref  PrimitiveTypeStringOfSystem()
    { 
      const static String aString( "System" );
      return aString;
    }

  private:

    Type theType;

  };

  /** @} */ //end of libecs_module 
  
} // namespace libecs


#endif /* __PRIMITIVETYPE_HPP */
