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

#ifndef __ENTITYTYPE_HPP 
#define __ENTITYTYPE_HPP 

#include "libecs.hpp"


namespace libecs
{

  /** @addtogroup identifier
   *
   *@{
   */

  /** @file */


  /**

  */

  class EntityType
  {

  public:
    
    enum Type
      {
	NONE      = 0,
	ENTITY    = 1,
	VARIABLE  = 2,
	PROCESS   = 3,
	SYSTEM    = 4
      };

    ECELL_API EntityType( StringCref typestring );

    EntityType( const int number );

    EntityType( const Type type )
      :
      theType( type )
    {
      ; // do nothing
    }

    EntityType( EntityTypeCref primitivetype )
      :
      theType( primitivetype.getType() )
    {
      ; // do nothing
    }

    EntityType()
      :
      theType( NONE )
    {
      ; // do nothing
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

    //    operator const int&() const
    //    {
    //      return static_cast<const int&>( getType() ); 
    //    }

    bool operator<( EntityTypeCref rhs ) const
    {
      return theType < rhs.getType();
    }

    bool operator<( const Type rhs ) const
    {
      return theType < rhs;
    }

    bool operator==( EntityTypeCref rhs ) const
    {
      return theType == rhs.getType();
    }

    bool operator==( const Type rhs ) const
    {
      return theType == rhs;
    }


    static StringCref  EntityTypeStringOfNone();

    static StringCref  EntityTypeStringOfEntity();

    static StringCref  EntityTypeStringOfProcess();
    
    static StringCref  EntityTypeStringOfVariable();
    
    static StringCref  EntityTypeStringOfSystem();

  private:

    Type theType;

  };

  /*@}*/

} // namespace libecs


#endif /* __ENTITYTYPE_HPP */
