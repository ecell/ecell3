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





#ifndef ___ENTITY_H___
#define ___ENTITY_H___
#include <cassert>
#include <string>
#include "util/Message.h"
#include "Defs.h"

class System;
class RootSystem;

/*!
Entity class is a base class for all components in the cell model.
Entity is also a MessageInterface. (i.e. It can accept Message objects
or request for Message.)

It has entryname, name and supersystem as common properties.
*/
class Entity : public MessageInterface
{
friend class RootSystem;


public: // MessageInterfaces.

  // none

private:

  System* _supersystem;
  string _entryname;
  string _name;

protected:

  virtual void makeSlots();

public:

  Entity(); 
  virtual ~Entity();

  /*!
    Set supersystem pointer of this Entity.  

    \param supersystem name of a System object to which this object
    belongs.
   */
//  virtual void setSupersystem(const string& supersystem);

  /*!
    Set supersystem pointer of this Entity.  
    Usually no need to set this manually because a System object will 
    do this when an Entity is installed to the System.

    \param supersystem name of a System object to which this object
    belongs.
   */
  virtual void setSupersystem(System* const supersystem) 
   {_supersystem = supersystem;}

  /*!
    Set entryname(identifier) of this Entity.

    \param entryname entryname of this Entry.
   */
  void setEntryname(const string& entryname) {_entryname = entryname;}

  /*!
    Set name of this Entity.

    \param name name of this Entity.
   */
  void setName(const string& name) {_name = name;}

  /*!
    \return entryname of this Entity.
   */
  const string& entryname() const {return _entryname;}

  /*!
    \return name of this Entity.
   */
  const string& name() const {return _name;}

  /*!
    \return System path of this Entity.
   */
  const string systemPath() const;

  /*!
    \return FQEN (Fully Qualified Entry Name) of this Entity.
   */
  const string fqen() const;

  /*!
    \return FQPN (Fully Qualified Primitive Name) of this Entity.
   */
  const string fqpn() const;

  /*!
    Returns activity value of this Entity defined in subclasses.
    Thus this should be overrided to calculate and return activity value 
    defined in each derived class. The time width used in this method
    should be delta-T. In case activity per second is needed, use
    activityPerSec() method.

    \return activity of this Entity
    \sa activityPerSec()
   */
  virtual Float activity();

  /*!
    Returns activity value (per second).
    Default action of this method is to return activity() / delta-T,
    but this action can be changed in subclasses.

    \return activity of this Entity per second
   */
  virtual Float activityPerSec();

  System* supersystem() const {return _supersystem;}

  virtual const char* const className() const { return "Entity"; }
};


typedef Entity* EntityPtr;

#endif /*  ___ENTITY_H___ */
