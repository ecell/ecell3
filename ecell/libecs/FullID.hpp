//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

#ifndef ___FULLID_H___
#define ___FULLID_H___
#include <string>

#include "libecs.hpp"
#include "PrimitiveType.hpp"


namespace libecs
{

  /** 
      SystemPath 
  */
  class SystemPath : public StringList
  {

  public:

    SystemPath( StringCref systempathstring = "" )
    {
      parse( systempathstring );
    }

    SystemPath( SystemPathCref systempath )
      :
      StringList( static_cast<StringList>( systempath ) )
    {
      ; // do nothing
    }

    ~SystemPath() {}

    const String getString() const;
    operator const String()  const  { return getString(); }

    bool isAbsolute() const
    {
      return front()[0] == DELIMITER;
    }

    bool isValid() const
    {
      // FIXME: check '..'s and '.'s etc..
      return true;
    }

  protected:

    /**
       Standardize a SystemPath. 
       Reduce '..'s and remove extra white spaces.

       @return reference to the systempath
    */
    void standardize();

  private:

    //    SystemPath();


    void parse( StringCref systempathstring );

  public:

    static const char DELIMITER = '/';

  };


  /**
     FullID is an identifier of a unique Entiy in a cell model.
     The FullID consists of a PrimitiveType, a SystemPath and an ID string.

     @see PrimitiveType, SystemPath
  */
  class FullID
  {

  public:

    FullID( const PrimitiveType type,
	    SystemPathCref systempath,
	    StringCref id )
      :
      thePrimitiveType( type ),
      theSystemPath( systempath ),
      theID( id )
    {
      ; // do nothing
    }

    FullID( StringCref fullidstring )
    {
      parse( fullidstring );
    }

    FullID( FullIDCref fullid )
      :
      thePrimitiveType( fullid.getPrimitiveType() ),
      theSystemPath( fullid.getSystemPath() ),
      theID( fullid.getID() )
    {
      ; // do nothing
    }


    ~FullID() {}
  
    const PrimitiveType  getPrimitiveType() const 
    { 
      return thePrimitiveType; 
    }

    const SystemPathCref getSystemPath()    const 
    { 
      return theSystemPath; 
    }

    const StringCref     getID()            const 
    { 
      return theID;
    }


    void setPrimitiveType( const PrimitiveType type )
    {
      thePrimitiveType = type;
    }

    void setSystemPath( SystemPathCref systempath ) 
    {
      theSystemPath = systempath;
    }

    void setID( StringCref id ) 
    {
      theID = id;
    }

    bool isValid() const;

    const String getString() const;
    operator const String()  const  { return getString(); }

    bool operator<( FullIDCref rhs ) const
    {
      // first look at the PrimitiveType
      if( getPrimitiveType() != rhs.getPrimitiveType() )
	{
	  return getPrimitiveType() < rhs.getPrimitiveType();
	}

      // then compare the SystemPaths
      // FIXME: should be faster is there is SystemPath::compare()
      if( getSystemPath() != rhs.getSystemPath() )
	{
	  return getSystemPath() < rhs.getSystemPath();
	}

      // finally compare the ID strings
      return getID() < rhs.getID();
    }

    bool operator==( FullIDCref rhs ) const
    {
      // first look at the PrimitiveType
      if( getPrimitiveType() != rhs.getPrimitiveType() )
	{
	  return false;
	}

      // then compare the SystemPaths
      if( getSystemPath() != rhs.getSystemPath() )
	{
	  return false;
	}

      // finally compare the ID strings
      return getID() == rhs.getID();
    }

  protected:

    void parse( StringCref fullidstring );

  private:

    FullID();

  public:

    static const char DELIMITER = ':';

  private:

    PrimitiveType thePrimitiveType;
    SystemPath    theSystemPath;
    String        theID;

  };

  class FullPropertyName
  {

  public:

    FullPropertyName( const PrimitiveType type, 
		      SystemPathCref systempath,
		      StringCref id,
		      StringCref propertyname )
      :
      theFullID( type, systempath, id ),
      thePropertyName( propertyname )
    {
      ; // do nothing
    }

    FullPropertyName( FullIDCref fullid, StringCref propertyname )
      :
      theFullID( fullid ),
      thePropertyName( propertyname )
    {
      ; // do nothing
    }

    //    FullPropertyName( FullPropertyNameCref fullpropertyname );

    FullPropertyName( StringCref fullpropertynamestring );

    ~FullPropertyName() {}



    const FullIDCref     getFullID()        const
    {
      return theFullID;
    }

    const PrimitiveType  getPrimitiveType() const 
    { 
      return getFullID().getPrimitiveType(); 
    }

    const SystemPathCref getSystemPath()    const
    { 
      return getFullID().getSystemPath();
    }

    const StringCref     getID()            const
    { 
      return getFullID().getID();
    }

    const StringCref     getPropertyName()  const
    {
      return thePropertyName;
    }

    void setPrimitiveType( const PrimitiveType type )
    {
      theFullID.setPrimitiveType( type );
    }

    void setSystemPath( SystemPathCref systempath ) 
    {
      theFullID.setSystemPath( systempath );
    }

    void setID( StringCref id ) 
    {
      theFullID.setID( id );
    }

    void setPropertyName( StringCref propertyname )
    {
      thePropertyName = propertyname;
    }

    const String getString() const;
    operator const String()  const  { return getString(); }

    bool isValid() const;

    bool operator<( FullPropertyNameCref rhs ) const
    {
      if( getFullID() != rhs.getFullID() )
	{
	  return getFullID() < rhs.getFullID();
	}

      return getPropertyName() < rhs.getPropertyName();
    }

    bool operator==( FullPropertyNameCref rhs ) const
    {
      if( getFullID() != rhs.getFullID() )
	{
	  return false;
	}

      // finally compare the ID strings
      return getPropertyName() == rhs.getPropertyName();
    }

  private:

    FullID theFullID;
    String thePropertyName;

  };



} // namespace libecs

#endif /*  ___FQPI_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
