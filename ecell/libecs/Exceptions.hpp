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

  virtual const char* what() const 
    { return "E-Cell Core System Standard Exception"; }

  virtual const String message() const 
    {
#ifdef DEBUG
      return theMethod + ":\n" + 
	String( getClassName() ) + ": " + what() + ": \n\t"
	+ theMessage + "\n";
#else
      return String( getClassName() ) + ": " + what() + ": \n\t"
	+ theMessage + "\n";
#endif /* DEBUG */
    }

  virtual const char* const getClassName() const  {return "Exception";}

protected:

  const String theMethod;
  const String theMessage;
};

#define DEFINE_EXCEPTION( CLASSNAME, BASECLASS, WHAT )\
class CLASSNAME : public BASECLASS\
{\
public:\
  CLASSNAME( StringCref method, StringCref message = "" )\
    :  BASECLASS( method, message ) {}\
  virtual ~CLASSNAME() {}\
  const char* what() const { return WHAT; }\
  virtual const char* const getClassName() const\
    { return #CLASSNAME ; }\
};\

DEFINE_EXCEPTION( UnexpectedError,       Exception, 
		  "Unexpected error" );
DEFINE_EXCEPTION( NotFound,              Exception,
		  "Not found" );
DEFINE_EXCEPTION( CantOpen,              Exception, 
		  "Can't open file" );
DEFINE_EXCEPTION( BadID,                 Exception, 
		  "Bad ID" );
DEFINE_EXCEPTION( MessageException,      Exception, 
		  "Message Exception" );
DEFINE_EXCEPTION( CallbackFailed,        Exception,
		  "Callback has Failed" );
DEFINE_EXCEPTION( BadMessage,            MessageException, 
		  "Bad Message" );
DEFINE_EXCEPTION( NoMethod,              MessageException, 
		  "No method registered for the slot" );
DEFINE_EXCEPTION( NoSlot,                MessageException, 
		  "No slot found for the message" );
DEFINE_EXCEPTION( InvalidPrimitiveType,  Exception,
		  "Invalid PrimitiveType" );


#endif /* ___EXCEPTIONS_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
