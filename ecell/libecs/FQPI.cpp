
char const FQPN_C_rcsid[] = "$Id$";
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




#include <string>
#include "FQPN.h"


///////////////////////  SystemPath

SystemPath::SystemPath(const string& rqsn) : _systempath(rqsn)
{
  standardize();

}


const string SystemPath::last() const
{
  int i = _systempath.rfind(DELIMITER); ++i;
  return _systempath.substr(i,string::npos);
}

const string SystemPath::first() const
{
 int i = _systempath.find(DELIMITER); 
 if(i != 0)
   return _systempath.substr(0,i);

 return "/";

// int j = _systempath.find(DELIMITER,1);
// return _systempath.substr(0,j-1);
}

SystemPath SystemPath::next() const 
{
  string::size_type i = _systempath.find(DELIMITER);
  if(i!=string::npos)
    {
      ++i;
      return SystemPath(_systempath.substr(i,string::npos)); 
    }
  else
    return SystemPath("");
}


void SystemPath::standardize()
{
  // FIXME: incomplete
   
}


////////////////////////////////  FQEN

FQEN::FQEN(const string& systemname,const string& entryname)
:SystemPath(systemname),_entryname(entryname)
{
  
}

FQEN::FQEN(const string& fqen) 
: SystemPath(systempathOf(fqen)),_entryname(entrynameOf(fqen))
{
  standardize();
}

const string FQEN::entrynameOf(const string& fqen)
{
  string::size_type border = fqen.find(':');
  if(border == string::npos)
    throw BadFQEN(__PRETTY_FUNCTION__,
		  "no \':\' found in \"" + fqen + "\".");
  if(fqen.find(':',border+1) != string::npos)
    throw BadFQEN(__PRETTY_FUNCTION__,
		  "too many \':\' in \"" + fqen + "\".");

  return fqen.substr(border+1,string::npos);
}

const string FQEN::systempathOf(const string& fqen)
{
  string::size_type border = fqen.find(':');
  if(border == string::npos)
    throw BadFQEN(__PRETTY_FUNCTION__,
		  "no \':\' found in \"" + fqen + "\".");
  if(fqen.find(':',border+1) != string::npos)
    throw BadFQEN(__PRETTY_FUNCTION__,
		  "to many \':\' in \"" + fqen + "\".");

  return fqen.substr(0,border);
}


const string FQEN::fqenString() const
{
  return (systemPathString() + ":" + entrynameString());
}


////////////////////////////////  FQPN

FQPN::FQPN(const Primitive::Type type,const FQEN& fqen)
: FQEN(fqen),_type(type)
{
}

FQPN::FQPN(const string& fqpn) : FQEN(fqenOf(fqpn)),_type(typeOf(fqpn))
{
}

const string FQPN::fqenOf(const string& fqpn)
{
  string::size_type border;
  border = fqpn.find(':');
  if(border == string::npos)
    throw BadFQPN(__PRETTY_FUNCTION__,
		  "no \':\' found in \"" + fqpn + "\".");
  if(fqpn.find(':',border+1) == string::npos)
        throw BadFQPN(__PRETTY_FUNCTION__,
		      "no enough \':\' found in \"" + fqpn + "\".");

  return fqpn.substr(border+1,string::npos);
}

Primitive::Type FQPN::typeOf(const string& fqpn)
{
  string::size_type border;
  border = fqpn.find(':');
  if(border == string::npos)
    throw BadFQPN(__PRETTY_FUNCTION__,
		  "no \':\' found in \"" + fqpn + "\".");
  if(fqpn.find(':',border+1) == string::npos)
        throw BadFQPN(__PRETTY_FUNCTION__,
		      "no enough \':\' found in \"" + fqpn + "\".");
  
  string typestring = fqpn.substr(0,border);
  return Primitive::PrimitiveType(typestring);
}


const string FQPN::fqpnString() const 
{
  return (Primitive::PrimitiveTypeString(_type) + ":" + fqenString());
}
