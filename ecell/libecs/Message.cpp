
char const Message_C_rcsid[] = "$Id$";
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




#include <strstream>
#include <stdio.h>
#include "Message.h"
#include "StringList.h"
//FIXME: #include "ecell/MessageWindow.h"

////////////////////// Message

Message::Message(const string& keyword,const string& body) 
{
  first = keyword;
  second = body;
}

Message::Message(const string& message) 
{
  string::size_type j = message.find(FIELD_SEPARATOR);
  if(j != string::npos)
    {
      first = message.substr(0,j);
      string::size_type k = message.find_first_not_of(FIELD_SEPARATOR,j);
      second = message.substr(k,string::npos);
    }
  else
    {
      first = message;
      second = "";
    }
//  cerr << "MESSAGE:" << first << "::" << second << endl;

}

Message::Message(const string& keyword,const Float f)
{
  first = keyword;
  ostrstream os;
  os.precision(FLOAT_DIG);
  os << f;
  os << ends;  // by naota on 29. Nov. 1999
  second = os.str();
}

Message::Message(const string& keyword,const Int i)
{
  first = keyword;
  ostrstream os;
  os << i;
  os << ends;  // by naota on 29. Nov. 1999
  second = os.str();
}

Message::Message(const Message& message)
{
  operator=(message);
}

Message& Message::operator=(const Message& rhs)
{
  if(this != &rhs)
    {
      first = rhs.keyword();
      second = rhs.body();
    }
  return *this;
}

Message::~Message()
{

}

const string Message::body(int n) const
{
  string::size_type pos = 0;
  while(n != 0)
    {
      pos = second.find(FIELD_SEPARATOR,pos);
      if(pos == string::npos)
	return "";
      --n;
      ++pos;
    }
  return second.substr(pos,second.find(FIELD_SEPARATOR)-pos);
}

//FIXME:

///////////////////////////// MessageInterface

MessageInterface::MessageInterface()
{
//  makeSlots();
}

MessageInterface::~MessageInterface()
{
  for(SlotMapIterator i = _slotMap.begin() ; i != _slotMap.end() ; i++)
    delete i->second;
}

void MessageInterface::appendSlot(const string& keyword, 
			      AbstractMessageCallback* func)
{
//  cerr << keyword << endl;
  if(_slotMap.find(keyword) != _slotMap.end())
    {
      //      *theMessageWindow << "MessageSlot: appendSlot(): slot for keyword [" 
      //	<< keyword << "] already exists. Taking later one.\n";
      delete _slotMap[keyword];
      _slotMap.erase(keyword);
    }
  _slotMap[keyword] = func;
}

void MessageInterface::deleteSlot(const string& keyword)
{
  if(_slotMap.find(keyword) == _slotMap.end())
    {
      //      *theMessageWindow << "MessageSlot: deleteSlot(): no slot for keyword [" 
      //	<< keyword << "] found.\n";
      return;
    }
  delete _slotMap[keyword];
  _slotMap.erase(keyword);
}

void MessageInterface::set(const Message& message) 
throw(NoSuchSlot)
{
  // debugDump();
  // cerr << message.dump() << endl;

  SlotMapIterator sm;
  if((sm = _slotMap.find(message.keyword())) == _slotMap.end())
    throw NoSuchSlot(__PRETTY_FUNCTION__,
		     className() + string(": got a Message (keyword = [")
		     + message.keyword() + "]) but no slot for it.");

  try{
    sm->second->set(message);
  }
  catch(Exception& e)
    {
      //      *theMessageWindow << className() << ": Callback has failed (keyword = [" 
      //	<< message.keyword() << "]):\n\t" << e.message() << "\n";
      return;
    }
  catch(...)
    {
      //      *theMessageWindow << __PRETTY_FUNCTION__ << ": " 
      //<< "callback has failed.(keyword = [" << message.keyword() << "])\n";
      return;
    }

}

const Message MessageInterface::get(const string& keyword) 
throw(NoSuchSlot)
{
  SlotMapIterator sm;
  if((sm = _slotMap.find(keyword)) == _slotMap.end())
    throw NoSuchSlot(__PRETTY_FUNCTION__,className()
		     + string(": got a request for Message (keyword = [")
		     + keyword + "] but no slot for it.\n");
  try {
    return sm->second->get(keyword);
  }
  catch(Exception& e)
    {
      //      *theMessageWindow << className() << ": Callback has failed (keyword = [" 
      //	<< keyword << "]):\n\t" << e.message() << "\n";
      return Message(keyword,"");
    }
  catch(...)
    {
      //      *theMessageWindow << __PRETTY_FUNCTION__ << ": " 
      //	<< "callback has failed.(keyword = [" << keyword << "])\n";
      return Message(keyword,"");
    }
}

StringList MessageInterface::slotList()
{
  StringList sl;
  for(SlotMapIterator i = _slotMap.begin() ; i != _slotMap.end() ; i++)
    sl.insert(sl.end(),i->first);
  return sl;
}


void debugPrint(FILE * pf, const string & str)
{
  for (string::size_type iii = 0; iii < str.size(); iii++) {
    int charCode = static_cast<int>(str.at(iii)) & 0x00ff;
    if (0x20 <= charCode && charCode <= 0x7e) {
      putc(charCode, pf);
    } else {
      fprintf(pf, "\\x%02X", charCode);
    }
  }
  putc('\n', pf);
}


void debugPrint(char * buf, const string & str)
{
  for (string::size_type iii = 0; iii < str.size(); iii++) {
    int charCode = static_cast<int>(str.at(iii)) & 0x00ff;
    if (0x20 <= charCode && charCode <= 0x7e) {
      *buf++ = static_cast<char>(charCode);
    } else {
      sprintf(buf, "\\x%02X", charCode);
      buf += 4;
    }
  }
  *buf = '\0';
}


void MessageInterface::debugDump() const
{
  char bufDebug1[1000], bufDebug2[1000];

  fprintf(stderr, "========  MessageInterface::debugDump() ========\n");
  for (SlotMap::const_iterator iii = _slotMap.begin(); iii != _slotMap.end(); iii++) {
   debugPrint(bufDebug1, iii->first);
   debugPrint(bufDebug2, iii->second->get(iii->first).body());
   fprintf(stderr, "  \"%s\", \"%s\"\n", bufDebug1, bufDebug2);
  }
  fprintf(stderr, "======== end of MessageInterface::debugDump() ========\n");
}
