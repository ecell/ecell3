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
#include <stdexcept>

#include "config.h"
#include "Defs.hpp"

/// Base exception class
class Exception : public exception
{
public:
  Exception( StringCref method, StringCref message = "" )
    : theMethod( method ), theMessage( message ) {}
  virtual ~Exception() {}

  virtual const String message() const 
    {
#ifdef DEBUG
      return theMethod + ":\n" + 
	String( getClassName() ) + ": " + theMessage + "\n";
#else
      return String( getClassName() ) + ": " + theMessage + "\n";
#endif /* DEBUG */
    }

  virtual const char* what() const { return theMessage.c_str(); }
  virtual const char* const getClassName() const  {return "Exception";}

protected:

  const String theMessage;
  const String theMethod;
};

#define DEFINE_EXCEPTION( CLASSNAME, BASECLASS )\
class CLASSNAME : public BASECLASS\
{\
public:\
  CLASSNAME( StringCref method, StringCref message = "" )\
    :  BASECLASS( method, message ) {}\
  virtual ~CLASSNAME() {}\
  virtual const char* const getClassName() const\
    { return #CLASSNAME ; }\
};\

DEFINE_EXCEPTION( UnexpectedError,       Exception);
DEFINE_EXCEPTION( NotFound,              Exception);
DEFINE_EXCEPTION( CantOpen,              Exception); 
DEFINE_EXCEPTION( BadID,                 Exception); 
DEFINE_EXCEPTION( MessageException,      Exception);
DEFINE_EXCEPTION( CallbackFailed,        Exception);
DEFINE_EXCEPTION( BadMessage,            MessageException); 
DEFINE_EXCEPTION( NoMethod,              MessageException);
DEFINE_EXCEPTION( NoSlot,                MessageException);
DEFINE_EXCEPTION( InvalidPrimitiveType,  Exception);

#endif /* ___EXCEPTIONS_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
