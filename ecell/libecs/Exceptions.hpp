//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER

#ifndef ___EXCEPTIONS_H___
#define ___EXCEPTIONS_H___
#include "config.h"
#include <typeinfo>
#include "util/Util.h"

#ifdef USE_LONGJMP
#include <setjmp.h>
extern jmp_buf jmpToMainLoop;
#include <signal.h>
#endif /* USE_LONGJMP */

//! Base exception class
class Exception 
{
protected:

  const string _method;
  const string _message;

public:
  Exception(const string& method,const string& message = "")
    : _method(method),_message(message) {}
  virtual ~Exception() {}

  virtual const string what() const 
    {return "E-Cell System Standard Exception";}

  virtual const string message() const 
    {
      return (string(className()) + ": " + what() + ": \n\t"
#ifdef DEBUG
	      + "occured at " + _method + ":\n\t" 
#endif /* DEBUG */
	      + _message + "\n");
    }

  virtual const char* const className() const 
    {return "Exception";}
//    {return decodeTypeName(typeid(*this)).c_str();}
};


class UnexpectedError : public Exception
{

public:

  UnexpectedError(const string& method,const string& message = ""):
  Exception(method,message) {}
  virtual ~UnexpectedError() {}

  const string what() const {return "Unexpected error";}
  virtual const char* const className() const 
    {return "UnexpectedError";}
};


class CantOpenFile : public Exception
{
  const string _filename;

public:
  CantOpenFile(const string& method,const string& message,
	       const string& filename)
    : Exception(method,message),_filename(filename) {}
  virtual ~CantOpenFile() {}

  const string what() const {return "Can't open file [" + _filename + "]";}
  virtual const char* const className() const 
    {return "CantOpenFile";}
};


class ClientMessageException : public Exception
{
public:
  ClientMessageException(const string & method, const string & message)
    : Exception(method, message) {};
  ClientMessageException(char const *pcMethod, char const * pcMessage)
    : Exception(pcMethod, pcMessage) {};
  virtual ~ClientMessageException() {};
  const string what() const {
    string tmp("UserNotify event");
    return tmp;
  }
  virtual const char* const className() const 
    {return "ClientMessageException";}
};


class CallRescueException : public Exception
{
public:
  CallRescueException(const string & method, const string & message)
    : Exception(method, message) {};
  CallRescueException(char const *pcMethod, char const * pcMessage)
    : Exception(pcMethod, pcMessage) {};
  virtual ~CallRescueException() {};
  const string what() const {
    string tmp("UserNotify event");
    return tmp;
  }
  virtual const char* const className() const 
    {return "CallRescueException";}
};


inline void call_rescue()
{
#ifdef USE_LONGJMP
  raise(SIGINT);
#else /* USE_LONGJMP */
  throw CallRescueException(__PRETTY_FUNCTION__, "call_rescue()");
#endif /* USE_LONGJMP */
}


#endif /* ___EXCEPTIONS_H___ */
