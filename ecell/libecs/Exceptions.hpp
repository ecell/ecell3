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

#ifndef ___EXCEPTIONS_H___
#define ___EXCEPTIONS_H___
#include <stdexcept>

#include "config.h"
#include "Defs.hpp"


namespace libecs
{

  /** @defgroup exception The Exceptions
      The exceptions.

      @ingroup libecs
      @{
  */ 
  

  /// A macro to throw an exception, with method name
#define THROW_EXCEPTION( CLASS, MESSAGE )\
throw CLASS( __PRETTY_FUNCTION__, MESSAGE )

#if defined( DEBUG ) 
#define DEBUG_EXCEPTION( EXPRESSION, CLASS, MESSAGE )\
if( ! ( EXPRESSION ) )\
{\
  THROW_EXCEPTION( CLASS, "[ " + String( STR( EXPRESSION )  ) + " ]\n\t"\
                   + MESSAGE );\
}
#else
#define DEBUG_EXCEPTION( EXPRESSION, CLASS, MESSAGE ) ;
#endif



  /// Base exception class
  class Exception 
    : public std::exception 
  {
  public:
    Exception( StringCref method, StringCref message = "" )
      : 
      theMethod( method ), 
      theMessage( message ) 
    {
      ; // do nothing
    }

    virtual ~Exception() throw() {}

    virtual const String message() const 
    {
#ifdef DEBUG
      return theMethod + ":\n" + 
	String( getClassName() ) + ": " + theMessage + "\n";
#else
      return String( getClassName() ) + ": " + theMessage + "\n";
#endif /* DEBUG */
    }

    virtual const char* what() const throw() { return message().c_str(); }
    virtual const char* const getClassName() const  { return "Exception"; }

  protected:

    const String theMethod;
    const String theMessage;
  };

  /**

  @internal
  */

#define DEFINE_EXCEPTION( CLASSNAME, BASECLASS )\
class CLASSNAME : public BASECLASS\
{\
public:\
  CLASSNAME( StringCref method, StringCref message = "" )\
    :  BASECLASS( method, message ) {}\
  virtual ~CLASSNAME() throw() {}\
  virtual const char* const getClassName() const\
    { return #CLASSNAME ; }\
};\


  // system errors
  DEFINE_EXCEPTION( UnexpectedError,       Exception);
  DEFINE_EXCEPTION( NotFound,              Exception);
  DEFINE_EXCEPTION( CantOpen,              Exception); 
  DEFINE_EXCEPTION( CallbackFailed,        Exception);
  DEFINE_EXCEPTION( AssertionFailed,       Exception);
  DEFINE_EXCEPTION( RangeError,            Exception);
  DEFINE_EXCEPTION( AlreadyExist,          Exception);

  // simulation errors
  DEFINE_EXCEPTION( InitializationFailed,  Exception);

  // PropertySlot errors
  DEFINE_EXCEPTION( PropertyException,     Exception);
  DEFINE_EXCEPTION( BadMessage,            PropertyException); 
  DEFINE_EXCEPTION( NoMethod,              PropertyException);
  DEFINE_EXCEPTION( NoSlot,                PropertyException);
  DEFINE_EXCEPTION( AttributeError,        PropertyException);

  // FullID errors
  DEFINE_EXCEPTION( BadID,                 Exception); 
  DEFINE_EXCEPTION( BadSystemPath,         BadID);
  DEFINE_EXCEPTION( InvalidEntityType,  BadID);


/**
   This macro throws UnexpectedError exception with a method name.

   Use this macro to indicate where must not be reached.
*/

#define NEVER_GET_HERE\
      THROW_EXCEPTION( UnexpectedError, \
		       "never get here (" + String( __PRETTY_FUNCTION__ )\
		       + ")." )


  /** @} */ //end of exception module

} // namespace libecs

#endif /* ___EXCEPTIONS_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
