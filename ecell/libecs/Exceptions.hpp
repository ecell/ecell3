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

#ifndef ___EXCEPTIONS_H___
#define ___EXCEPTIONS_H___
#include "config.h"
#include <typeinfo>
#include "include/Defs.hpp"
#include "util/Util.hpp"

DECLARE_CLASS( Exception );
DECLARE_CLASS( UnexpectedError );
DECLARE_CLASS( CantOpenFile );


/// Base exception class
class Exception 
{
public:
  Exception( StringCref method, StringCref message = "" )
    : theMethod( method ), theMessage( message ) {}
  virtual ~Exception() {}

  virtual const String what() const 
    { return "E-Cell Core System Standard Exception"; }

  virtual const String message() const 
    {
      return String( className() ) + ": " + what() + ": \n\t"
	      + theMessage + "\n";
    }

  virtual const char* const className() const  {return "Exception";}
//    {return decodeTypeName(typeid(*this)).c_str();}

protected:

  const String theMethod;
  const String theMessage;
};


class UnexpectedError : public Exception
{

public:

  UnexpectedError( StringCref method, StringCref message = "" )
    :  Exception( method, message ) {}
  virtual ~UnexpectedError() {}

  const String what() const { return "Unexpected error"; }
  virtual const char* const className() const 
    { return "UnexpectedError"; }
};


class CantOpenFile : public Exception
{

public:

  CantOpenFile( StringCref method, StringCref message,
		StringCref filename )
    : Exception( method, message ), theFilename( filename ) {}
  virtual ~CantOpenFile() {}

  const String what() const { return "Can't open file [" + theFilename + "]"; }
  virtual const char* const className() const 
    { return "CantOpenFile"; }

private:

  const String theFilename;

};


#endif /* ___EXCEPTIONS_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
