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





#ifndef ___FQPN_H___
#define ___FQPN_H___
#include <string>

#include "ecell/Exceptions.h"
#include "ecscore/Primitive.h"

/*! 
  SystemPath 
  */
class SystemPath {
public:

  static const char DELIMITER = '/';


  // exceptions.

  class SystemPathException : public Exception
    { 
    public: 
      SystemPathException(const string& method,const string& what) 
	: Exception(method,what) {} 
      const string what() const {return "";}
    };
  class BadSystemPath : public SystemPathException
    { 
    public: 
      BadSystemPath(const string& method,const string& what) 
	: SystemPathException(method,what) {} 
      const string what() const {return "Bad SystemPath.";}
    };


private:


protected:

  /*!
    Standardize a SystemPath. (i.e. convert RQSN -> FQSN)
    Reduce '..'s and remove trailing white spaces.

    \return reference to the systempath
    */
  void standardize();

  const string _systempath;
  SystemPath() {}

public:

  SystemPath(const string& systempath);
  virtual ~SystemPath() {}

  const string& systemPathString() const {return _systempath;}
//  virtual const string string() const {return systemPathString();}

  virtual operator string() const {return systemPathString();}

  /*!
    Extract the first system name. Implicitly standardize given string.
    \return Pointer to static char[]. Valid until next call.
    */
  const string first() const;
  /*!
    Extract the last system name. Implicitly standardize given string.

    \return Pointer to the last system name in given systempath.
    */
  const string last() const;
  /*!
    Remove the first system name. Implicitly standardize given string.
    \return
    */
  SystemPath next() const;

};

/*!
  FQEN(Fully Qualified EntryName)

  The Entryname is a identifier of Entity objects.  Given a Primitive type,
  one can identify unique Entity in RootSystem with a SystemPath and an entryname.
  \sa SystemPath, Primitive
*/
class FQEN : public SystemPath
{
public: // exceptions

  class FQENException : public Exception
    { 
    public: 
      FQENException(const string& method,const string& what)
	: Exception(method,what) {} 
      const string what() const {return "";}
    };
  class BadFQEN : public FQENException
    { 
    public: 
      BadFQEN(const string& method,const string& what) 
	: FQENException(method,what) {} 
      const string what() const {return "Bad FQEN";}
    };

private:

  const string _entryname;

public:

  FQEN(const string& systemname,const string& entryname);
  
  static const string entrynameOf(const string& fqen);
  static const string systempathOf(const string& fqen);

  FQEN(const string& fqen);
  virtual ~FQEN() {}

  const string fqenString() const;
  const string& entrynameString() const {return _entryname;}
//  virtual const string string() const {return fqenString();}
  virtual operator string() const {return fqenString();}
};

/*!
  FQPN (Fully Qualified Primitive Name).

  One can identify an unique Entiy in RootSystem with FQPN.
  The FQPN consists of FQEN and PrimitiveType.

  \sa FQEN, PrimitiveType
*/
class FQPN : public FQEN
{
  class FQPNException : public Exception
    { 
    public: 
      FQPNException(const string& method,const string& message) 
	: Exception(method,message) {} 
      const string what() const {return "";}
    };
  class BadFQPN : public FQPNException
    { 
    public:
      BadFQPN(const string& method,const string& message) 
	: FQPNException(method,message) {} 
      const string what() const {return "Bad FQPN.";}
    };

  Primitive::Type _type;

public:

  FQPN(const Primitive::Type type,const FQEN& fqen);
  FQPN(const string& fqpn);
  virtual ~FQPN() {}
  
  static const string fqenOf(const string& fqpn);
  static Primitive::Type typeOf(const string& fqpn);

  const Primitive::Type& type() const {return _type;}
  const string fqpnString() const;
//  virtual const string string() const {return fqpnString();}
  virtual operator string() const {return fqpnString();}
};

#endif /*  ___FQPN_H___ */
