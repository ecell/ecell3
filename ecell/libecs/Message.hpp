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





#ifndef ___MESSAGE_H___
#define ___MESSAGE_H___
#include <string>
#include <map>
#include <vector>
#include "Exceptions.h"
#include "Defs.h"
#include "StringList.h"

class AbstractMessageCallback;

typedef pair<string,string> StringPair;

class Message : private StringPair
{

public: // exceptions

  class MessageException : public Exception
    {
    public:
      MessageException(const string& method,const string& what)
	: Exception(method,what){}
    };
  class BadMessage : public MessageException
    {
    public:
      BadMessage(const string& method,const string& what)
	: MessageException(method,what){}
    };

private:

  // empty

public:

  Message(const string& keyword,const string& body); 
  Message(const string& keyword,const Float f);
  Message(const string& keyword,const Int i);
  Message(const string& message); 

  // copy procedures
  Message(const Message& message);
  Message& operator=(const Message&);
  
  virtual ~Message();

  /*!
    Returns keyword string of this Message.

    \return keyword string.
    \sa body()
   */
  const string& keyword() const {return first;}

  /*!
    Returns body string of this Message.

    \return body string.
    \sa keyword()
   */
  const string& body() const {return second;}

  /*!
    Returns nth field of body string using FIELD_SEPARATOR as delimiter.  

    \return nth field of body string.
    \sa FIELD_SEPARATOR
   */
  const string body(int n) const;


  const string dump() const {return first + ' ' + second;}
//  Float asFloat() const;
//  int asInt() const;
};


class AbstractMessageCallback
{

public:

  // exceptions

  class CallbackFailed : public Message::MessageException
    {
    public: 
      CallbackFailed(const string& method,const string& message)
	: MessageException(method,message) {}
      const string what() const {return "Callback has failed.";}
    };

  class NoMethod : public Message::MessageException
    {
    public: 
      NoMethod(const string& method,const string& what)
	: MessageException(method,what){}
      const string what() const {return "No method registered for the slot";}
    };

  class NoSetMethod : public NoMethod
    {
    public: 
      NoSetMethod(const string& method,const string& what)
	:NoMethod(method,what){}
      const string what() const {return "No set method registered for the slot";}
    };

  class NoGetMethod : public NoMethod
    {
    public: 
      NoGetMethod(const string& method,const string& what)
	:NoMethod(method,what){}
      const string what() const {return "No get method registered for the slot";}
    };


  virtual void set(const Message& message) =0;
  virtual const Message get(const string& keyword) =0;

  virtual void operator()(const Message& message) 
    {set(message);}
  virtual const Message operator()(const string& keyword) 
    {return get(keyword);}
};


template <class T>
class MessageCallback : public AbstractMessageCallback
{

  typedef void (T::*SetMessageFunc)(const Message&);
  typedef const Message (T::*GetMessageFunc)(const string&);

private:

  T& _obj;
//  Message::Type _type;
  const SetMessageFunc _setMethod;
  const GetMessageFunc _getMethod;

public:

  MessageCallback(T& obj,const SetMessageFunc set,const GetMessageFunc get)
    : _obj(obj),_setMethod(set),_getMethod(get) {}
  
  virtual void set(const Message& message) 
    {
//      if(!_setMethod)
//	throw NoSetMethod(__PRETTY_FUNCTION__,
//			  "no method to set [" + message.keyword()
//			  + "] found.");
      if(!_setMethod)
	return;
      (_obj.*_setMethod)(message);

    }

  virtual const Message get(const string& keyword)     
    {
//      if(!_getMethod)
//	throw NoGetMethod(__PRETTY_FUNCTION__,
//			  "no method to get [" + keyword
//			  + "] found.");
      if(!_getMethod)
	return Message(keyword,"");
      return ((_obj.*_getMethod)(keyword));
    }
};



/*!
  Common base class for classes which receive Messages.

  NOTE:  Subclasses of MessageInterface MUST call their own makeSlots() 
         to make slots unique to them (if any) in their constructors.
         (virtual functions doesn't work in constructors)

*/
class MessageInterface
{
public:  // exceptions

  class NoSuchSlot : public Message::MessageException
    {
    public: 
      NoSuchSlot(const string& method,const string& what)
	: MessageException(method,what){}
      const string what() const {return "No appropriate slot found";}
    };

private:

  typedef map<const string, AbstractMessageCallback*> SlotMap;
  typedef SlotMap::iterator SlotMapIterator;

  SlotMap _slotMap;

protected:

  void appendSlot(const string& keyword, AbstractMessageCallback*);
  void deleteSlot(const string& keyword);

public:

  MessageInterface();


  virtual ~MessageInterface();

  void set(const Message&) throw(NoSuchSlot);
  const Message get(const string&) throw(NoSuchSlot);
  StringList slotList();

  virtual void makeSlots()=0;

  virtual const char* const className() const {return "MessageInterface";}
  void debugDump() const;
};


#define MessageSlot(KEY,CLASS,OBJ,SETMETHOD,GETMETHOD)\
appendSlot(KEY,new MessageCallback<CLASS>\
	   (OBJ,static_cast<MessageCallback<CLASS>::SetMessageFunc>\
	    (SETMETHOD),\
	    static_cast<MessageCallback<CLASS>::GetMessageFunc>\
	    (GETMETHOD)))

#endif /* ___MESSAGE_H___*/


