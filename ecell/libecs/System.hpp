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





#ifndef ___SYSTEM_H___
#define ___SYSTEM_H___
#include <iostream.h>
#include <stl.h>
#include <string.h>

#include "ecscore/Entity.h"
#include "ecscore/Primitive.h"
#include "ecell/Exceptions.h"
#include "ecscore/FQPN.h"

class Gene;
class Genome;
class SystemMaker;
class GenomicElement;
class Stepper;
class Message;

// Tree data structures used for entry lists
typedef map<const string,Substance*> SubstanceList;
typedef map<const string,Reactor*>   ReactorList;
typedef map<const string,System*>    SystemList;
typedef list<GenomicElement*>        GenomicElementList;

// Iterator types 
typedef SubstanceList::iterator           SubstanceListIterator;
typedef ReactorList::iterator             ReactorListIterator;
typedef SystemList::iterator              SystemListIterator;
typedef GenomicElementList::iterator      GenomicElementListIterator;


class System;
typedef System (*SystemAllocatorFunc)();

//! A type for function that gets pointer to a Primitive and data.
typedef void (*PrimitiveCallback) (Primitive*,void* clientData);


#define isSSystem(S)    (typeid(S) == typeid(SSystem))
#define isRSystem(S)    (typeid(S) == typeid(SSystem))
#define isMetaSystem(S) (typeid(S) == typeid(MetaSystem))
#define isGenome(S)     (typeid(S) == typeid(Genome))


// print message when it is not expected class.
/*
#define INVALIDCLASS(FUNC,CLASS,RET)\
  *theMessageWindow << "error: FUNC request for [" << entryname() << \
    "], which is not CLASS .\n" ; return RET;
*/

#define INVALIDCLASS(FUNC,CLASS,RET)\
return false;


class System : public virtual Entity
{
friend class RootSystem;
friend class SystemMaker;

public: // exceptions

  class SystemException : public Exception
    {
    public:
      SystemException(const string& method,const string& message) 
	: Exception(method,message) {}
    };
  class InvalidPrimitiveType : public SystemException
    {
    public:
      InvalidPrimitiveType(const string& method,const string& message)
	: SystemException(method,message) {}
      const string what() const {return "Invalid Primitive type requested";}
    };
  class UnmatchedSystemClass : public SystemException
    {
    public:
      UnmatchedSystemClass(const string& method,const string& message) 
	: SystemException(method,message) {}
      const string what() const 
	{return "Primitive type and System class unmatched";}
    };
  class NotFound : public SystemException
    {
    public:
      NotFound(const string& method,const string& message) 
	: SystemException(method,message) {}
      const string what() const {return "Couldn't find requested Primitive";}
    };


public: // message interfaces

  void setStepper(const Message& message);
  const Message getStepper(const string& keyword);

  void setVolumeIndex(const Message& message);
  const Message getVolumeIndex(const string& keyword);

protected:

  virtual void makeSlots();

  FQEN* _volumeIndexName;

  Reactor* _volumeIndex;
  Stepper* _stepper;

public:

  System();
  virtual ~System();

  const string fqpn() const;

  virtual void initialize();

  virtual void clear(){}
  virtual void react(){}
  virtual void turn() {}
  virtual void transit(){}
  virtual void postern() {}

  virtual const char* const className() const { return "System"; }

  /*!
    \return An pointer to a Stepper object that this System has or\
    NULL pointer if it is not set.
    */
  Stepper* stepper() {return _stepper;}


  /*!
   Find and get a Primitive of given type and entryname.
   The Primitive object must be new'd and deleted by 
   caller of this method.    
 
   \param entryname Entryname of the Primitive to be obtained.
   \param primitive Pointer to a primitive whose type field is set\
   to a type of primitive to be obtained. 
 
   \return true -> success, false -> failed.
   */
  Primitive getPrimitive(const string& entryname,const Primitive::Type type)
    throw(UnmatchedSystemClass,InvalidPrimitiveType,NotFound);

  /*!
    Calls function of type PrimitiveCallback for each Entity of
    given type.

 
   \param type Type of Primitive. 
   \param cb A pointer to a function to be called for every Primitives.
   \param clientData void* pointer which is passed to the function.
 
   \return true -> success, false -> failed.
   \sa PrimitiveCallback
   */
  void forAllPrimitives(Primitive::Type type, PrimitiveCallback cb,
			void* clientData);

  /*!
    Returns the number of Entity of given type
    that this system have. 
 
   \param type Type of primitive.
 
   \return The number of instance of given type.\
   -1 if the type given is invalid.  0 if this is not a System\
   for given type.
   */
  int sizeOfPrimitiveList(Primitive::Type type);


  /*!
    Instantiate a Stepper object of \a classname using theRootSystem's
    StepperMaker object.  Register the Stepper object as a stepper for 
    this System.

    \param classname Classname of the Stepper that this System may have.
    */
  void setStepper(const string& classname);

  /*!
    This method takes a FQEN of a Reactor as a VolumeIndex of this System.
    The FQEN will be used to get a pointer to the VolumeIndex Reactor
    in initialize().

    \param fqen FQEN of a VolumeIndex Reactor for this System.
   */
  void setVolumeIndex(const FQEN& fqen);

  /*!
    \return a pointer to VolumeIndex Reactor of this System.
   */
  Reactor* volumeIndex() {return _volumeIndex;}

  /*!
    Volume of a System is calculated by activity() of
    VolumeIndex Reactor of the System.

    \return Volume of this System. Unit is [L].
   */
  virtual Float volume();

};

class RSystem : public virtual System
{
friend class RootSystem;


public:
  
  class isRegularReactorItem;

private:

  ReactorList _reactorList;


protected:

  ReactorListIterator _firstRegularReactor;

  virtual void initialize();
  virtual void clear() {}
  virtual void react();
  virtual void transit();
  virtual void postern();

public:

  RSystem();
  virtual ~RSystem() {}


   /*!
    Add a Reactor object in this RSystem.

    \return true -> success, false -> failed.
    */
  virtual bool newReactor(Reactor *const newone);
  
  /*!
    \return true: if this System contains a Reactor whose name is \a entryname.
    */
  bool RSystem::containsReactor(const string& entryname)
    {
      return (getReactorIterator(entryname) != _reactorList.end()) ? true : false;
    }

  /*!
    \return An iterator which points to the first Reactor in this System.
    */
  ReactorListIterator RSystem::firstReactor()
    {
      return _reactorList.begin();
    }

  /*!
    \return An iterator which points to the first regular Reactor\
    (i.e. not posterior Reactor) in this System.
    */
  ReactorListIterator RSystem::firstRegularReactor() const
    {
      return _firstRegularReactor;
    }

   /*!
    \return An iterator which points to the last Reactor in this System.
    */
  ReactorListIterator RSystem::lastReactor() 
    { 
      return _reactorList.end();
    }

   /*!
    \return An iterator which points to a Reactor whose name is \a entryname.
    */
  ReactorListIterator RSystem::getReactorIterator(const string& entryname)
    {
      return _reactorList.find(entryname);
    }

   /*!
    \return The number of Reactors in this object.
    */
  int RSystem::sizeOfReactorList() const
    {
      return _reactorList.size();
    }

  /*!
    \return An pointer to a Reactor object in this System named \a entryname.
    */
  Reactor* RSystem::getReactor(const string& entryname) throw(NotFound);

};


class SSystem : public virtual System
{
friend class RootSystem;

  SubstanceList _substanceList;


protected:

  virtual void initialize();
  virtual void clear();
  virtual void react() {}
  virtual void turn();
  virtual void transit();

public:

  SSystem();
  virtual ~SSystem() {}


  /*!
    Fixes a Substance object named \a entryname in this System.
    (i.e. bypass numerical integration.)
    */
  void fixSubstance(const string& entryname);

  /*!
    Add a Substance object in this SSystem.

    \return true -> success, false -> failed.
    */
  bool newSubstance(Substance* entryname);
  
  /*!
    \return An iterator which points to the first Substance in this System.
    */
  SubstanceListIterator SSystem::firstSubstance()
    {
      return _substanceList.begin();
    }

   /*!
    \return An iterator which points to the last Substance in this System.
    */
  SubstanceListIterator SSystem::lastSubstance()
    {
      return _substanceList.end();
    }

  /*!
    \return An iterator which points to a Substance whose name is \a entryname.
    */
  SubstanceListIterator SSystem::getSubstanceIterator(const string& entryname)
    {
      return _substanceList.find(entryname);
    }

  /*!
    \return true: if this System contains a Substance whose name is \a entryname.
    */
  bool SSystem::containsSubstance(const string& entryname)
    {
      return (getSubstanceIterator(entryname) != _substanceList.end()) ?
	true : false;
    }

   /*!
    \return The number of Substances in this object.
    */
  int SSystem::sizeOfSubstanceList() const
    {
      return _substanceList.size();
    }

  /*!
    \return An pointer to a Substance object in this System named \a entryname.
    */
  Substance* SSystem::getSubstance(const string& entryname) throw(NotFound);
};


//! MetaSystem is a system which can contain the other systems
class MetaSystem : public virtual System
{
public:

friend class RootSystem;

private:

  SystemList _subsystemList;


protected:

  virtual void initialize();

public:

  MetaSystem();
  virtual ~MetaSystem() {}

  /*!
    Add a System object in this MetaSystem

    \return true -> success, false -> failed.
    */
  virtual bool newSystem(System*);



  /*!
    \return An iterator which points to the first System in this System.
    */
  SystemListIterator MetaSystem::firstSystem()
    {
      return _subsystemList.begin();
    }

  /*!
    \return An iterator which points to the last System in this System.
    */
  SystemListIterator MetaSystem::lastSystem()
    {
      return _subsystemList.end();
    }

  /*!
    \return An iterator which points to a System whose name is \a entryname.
    */
  SystemListIterator MetaSystem::getSystemIterator(const string& entryname)
    {
      return _subsystemList.find(entryname);
    }

  /*!
    \return true: if this System contains a System whose name is \a entryname.
    */
  bool MetaSystem::containsSystem(const string& entryname)
    {
      return (getSystemIterator(entryname) != _subsystemList.end()) ? 
	true : false;
    }

  /*!
    \return The number of Systems in this object.
    */
  int MetaSystem::sizeOfSystemList() const
    {
      return _subsystemList.size();
    }

  /*!
    \return An pointer to a System object in this System named \a entryname.
    */
  System* MetaSystem::getSystem(const string& entryname) throw(NotFound);

  /*!
    This method finds recursively a System object pointed by
    \a systempath.

    \return An pointer to a System object in this or subsystems of this
    System object pointed by \a systempath
    */
  System* findSystem(const SystemPath& systempath)
    throw(NotFound); 



};



#include "ecscore/Reactor.h"
/*!
  Equivalent to Reactor::isRegularReactor except that
  this function object takes a reference to a ReactorList::value_type.
  */
class RSystem::isRegularReactorItem
: public unary_function<const ReactorList::value_type,bool>
{
public:
  bool operator() (const ReactorList::value_type r) const
    {
      return Reactor::isRegularReactor::isRegularName((r.second)->entryname());
    }
};




#endif /* ___SYSTEM_H___ */


