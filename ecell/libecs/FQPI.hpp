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

#ifndef ___FQPN_H___
#define ___FQPN_H___
#include <string>

#include "Exceptions.hpp"
#include "Primitive.hpp"

/** 
  SystemPath 
  */
class SystemPath {

public:

  static const char DELIMITER = '/';

  // exceptions.

  class SystemPathException : public Exception
    { 
    public: 
      SystemPathException( StringCref method, StringCref what ) 
	: Exception( method, what ) {} 
      const String what() const { return ""; }
    };
  class BadSystemPath : public SystemPathException
    { 
    public: 
      BadSystemPath( StringCref method, StringCref what ) 
	: SystemPathException( method, what ) {} 
      const String what() const { return "Bad SystemPath."; }
    };

public:

  SystemPath( StringCref systempath = "" );
  virtual ~SystemPath() {}

  StringCref getSystemPath() const { return theSystemPath; }
  virtual const String getString() const { return getSystemPath(); }

  virtual operator String() const { return getString(); }

  /**
    Extract the first system name. Standardize given string.
    @return name of the first system
    */
  const String first() const;

  /**
    Extract the last system name. Standardize given string.

    @return name of the last system in given systempath.
    */
  const String last() const;
  /**
    Remove the first system name. Standardize given string.
    @return
    */
  SystemPath next() const;

protected:

  /**
    Standardize a SystemPath. (i.e. convert RQSN -> FQSN)
    Reduce '..'s and remove trailing white spaces.

    @return reference to the systempath
    */
  void standardize();

  SystemPath() {}

private:

  const String theSystemPath;

};

/**
  FQIN(Fully Qualified Id Name)

  The Entryname is a identifier (ID) of Entity objects.  Given a
  Primitive type, one can identify unique Entity in a cell model with a
  SystemPath and an id.  
  @see SystemPath, Primitive 
*/
class FQIN : public SystemPath
{
public: // exceptions

  class FQINException : public Exception
    { 
    public: 
      FQINException( StringCref method, StringCref what )
	: Exception( method, what ) {} 
      const String what() const { return ""; }
    };

  class BadFQIN : public FQINException
    { 
    public: 
      BadFQIN( StringCref method, StringCref what ) 
	: FQINException( method,what ) {} 
      const String what() const { return "Bad FQIN"; }
    };

public:

  FQIN( StringCref systemname, StringCref id );
  FQIN( StringCref fqen );
  virtual ~FQIN() {}

  const String getFqin() const;
  virtual const String getString() const { return getFqin(); }
  StringCref getId() const { return theId; }
  virtual operator String() const { return getString(); }

  static const String IdOf( StringCref fqen );
  static const String SystemPathOf( StringCref fqen );

private:

  const String theId;

};

/**
  FQPN (Fully Qualified Primitive Name).

  One can identify an unique Entiy in a cell model with a FQPN.
  The FQPN consists of FQIN and PrimitiveType.

  @see FQIN, PrimitiveType
*/
class FQPN : public FQIN
{

public: // exceptions

  class FQPNException : public Exception
    { 
    public: 
      FQPNException( StringCref method, StringCref message ) 
	: Exception( method, message ) {} 
      const String what() const { return ""; }
    };
  class BadFQPN : public FQPNException
    { 
    public:
      BadFQPN( StringCref method, StringCref message ) 
	: FQPNException( method, message ) {} 
      const String what() const { return "Bad FQPN."; }
    };

public:

  FQPN( const Primitive::Type type, const FQIN& fqin );
  FQPN( StringCref fqpn );
  virtual ~FQPN() {}
  
  const String getFqpn() const;
  const Primitive::Type& getType() const { return theType; }

  virtual const String getString() const { return getFqpn(); }
  virtual operator String() const { return getString(); }

  static const String fqinOf( StringCref fqpn );
  static Primitive::Type typeOf( StringCref fqpn );

private:

  Primitive::Type theType;

};

#endif /*  ___FQPN_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
