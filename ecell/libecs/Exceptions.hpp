//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

#ifndef __EXCEPTIONS_HPP
#define __EXCEPTIONS_HPP

#include <stdexcept>
#include "libecs.hpp"

namespace libecs
{

/** @defgroup exception The Exceptions
    The exceptions.

    @ingroup libecs
    @{
*/

/// A macro to throw an exception, with method name
#define THROW_EXCEPTION( CLASS, MESSAGE )\
    throwException( CLASS( __PRETTY_FUNCTION__, MESSAGE ) )

/// Base exception class
class LIBECS_API Exception
            :
            public std::exception
{

public:
    Exception( const String& method, const String& message = "" )
            :
            theMethod( method ),
            theMessage( message )
    {
        ; // do nothing
    }

    virtual ~Exception() throw();

    const String& getMethod() const
    {
        return theMethod;
    }

    const String& getMessage() const
    {
        return theMessage;
    }

    virtual const char* what() const throw()
    {
        return asString().c_str();
    }

    virtual const char* const getClassName() const
    {
        return "Exception";
    }

    const String& asString() const;

    void swap( Exception& rhs )
    {
        theMethod.swap( rhs.theMethod );
        theMessage.swap( rhs.theMessage );
        theStringRepr.swap( rhs.theStringRepr );
    }
protected:
    String theMethod;
    String theMessage;
    String theStringRepr;
};

/**

@internal
*/

#define DEFINE_EXCEPTION( CLASSNAME, BASECLASS )\
class LIBECS_API CLASSNAME : public BASECLASS\
{\
public:\
  CLASSNAME( const String& method, const String& message = "" )\
    :  BASECLASS( method, message ) {}\
  virtual ~CLASSNAME() throw() {}\
  virtual const char* const getClassName() const\
    { return #CLASSNAME ; }\
};\
 

// system errors
DEFINE_EXCEPTION( UnexpectedError,        Exception );
DEFINE_EXCEPTION( NotFound,               Exception );
DEFINE_EXCEPTION( IOException,            Exception );
DEFINE_EXCEPTION( NotImplemented,         Exception );
DEFINE_EXCEPTION( AssertionFailed,        Exception );
DEFINE_EXCEPTION( AlreadyExist,           Exception );
DEFINE_EXCEPTION( ValueError,             Exception );
DEFINE_EXCEPTION( TypeError,              Exception );
DEFINE_EXCEPTION( OutOfRange,             Exception );
DEFINE_EXCEPTION( IllegalOperation,       Exception );
DEFINE_EXCEPTION( InitializationFailed,   Exception );
DEFINE_EXCEPTION( SimulationError,        Exception );
DEFINE_EXCEPTION( BadFormat,              Exception );
DEFINE_EXCEPTION( NoSlot,                 Exception );
DEFINE_EXCEPTION( Interruption,           Exception );

/**
   This macro throws UnexpectedError exception with a method name.

   Use this macro to indicate where must not be reached.
*/

#define NEVER_GET_HERE\
      THROW_EXCEPTION( libecs::UnexpectedError, \
         "never get here (" + libecs::String( __PRETTY_FUNCTION__ )\
         + ")." )

inline void throwException( const Exception& exc )
{
    throw exc;
}
} // namespace libecs

/** @} */ //end of exception module


#endif /* __EXCEPTIONS_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
