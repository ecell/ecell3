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





#ifndef ___REACTOR_H___
#define ___REACTOR_H___

#ifdef HAVE_LIMITS
#include <limits>
#else /* HAVE_LIMITS */
#include <climits>
#endif /* HAVE_LIMITS */

#include <stl.h>
#include "Defs.h"
#include "util/Util.h"
#include "Entity.h"
#include "util/Message.h"

class Substance;
class System;
class RSystem;
class Reactant;

class FQEN;

typedef vector<Reactant*> ReactantList;
typedef vector<Reactant*>::iterator ReactantListIterator;


/*!
  Reactor class is used to represent chemical and other phenonema
  which result in change in quantity of one or more Substances.
*/
class Reactor : public Entity
{
friend class Rule;
friend class RuleInterpreter;
friend class ReactorMaker;
friend class StandardRule;
friend class RSystem;
  

public:

/*!
  An predicate object which returns true if given pointer of Reactor 
  points to a regular Reactor (i.e. not posterior Reactor).
  
  Entryname of Posterior Reactors must start with '!'.
  */
class isRegularReactor : public unary_function<const Reactor*,bool>
{
public:
  static bool isRegularName(const string& entryname)
    {
      if(entryname[0] != '!')
	return true;
      return false;
    }
  bool operator() (const Reactor* r) const
    {
      return isRegularName(r->entryname());
    }
};



public: // MessageInterfaces

  void setSubstrate(const Message& message);
  void setProduct(const Message& message);
  void setCatalyst(const Message& message);
  void setEffector(const Message& message);
  void setInitialActivity(const Message& message);

  void setSubstrate(const FQEN& fqen,int coefficient);
  void setProduct(const FQEN& fqen,int coefficient);
  void setCatalyst(const FQEN& fqen,int coefficient);
  void setEffector(const FQEN& fqen,int coefficient);
  void setInitialActivity(Float activity);

  const Message getInitialActivity(const string& keyword);


  /*!
    Reactor Condition enum is used by Reactor's self-diagnosis system.

    There are two types of Reactor conditions, local and global.
    Local Reactor condition accessed by status() method indicates
    condition of the Reactor.
    Global Reactor condition can be checked by globalCondition() method
    and it returns Bad if there is at least one Reactor in the RootSystem
    which is in condition other than Good.

    /sa status() globalCondition()
    */
  enum Condition {
    /*! Condition Good. Everything seems to be Ok. */ 
    Good = 0x00,
    /*! Something wrong was occured at the time of initialization. */
    InitFail = 0x01,
    /*! Failed to do something at the time of react phase.  */
    ReactFail = 0x02,
    /*! Something is wrong, but don't know what is it. */
    Bad= 0x04,
    /*! Fatal.  Cannot continue. */
    Fatal= 0x08,
    /*! This Reactor is premature. Still uninitialized or need
     some more configuration to work correctly. */
    Premature= 0x16
    };

  enum LigandType {Substrate,Product,Catalyst,Effector};
  static const char* LIGAND_STRING_TABLE[];

private:

  Float _initialActivity;
  Float _activityBuffer;
  Condition _condition;
  static Condition _globalCondition;

protected:

  Float _activity;

  void makeSlots();

  ReactantList _substrateList; 
  ReactantList _productList;   
  ReactantList _effectorList;
  ReactantList _catalystList;

  /*!
    Set activity variable.  This must be set at every turn and takes
    [number of molecule that this reactor yields] / [deltaT].
    However, public activity() method returns it as number of molecule
    per a second, not per deltaT.

    \param activity [number of molecule that this yields] / [deltaT].
    \sa activity()
   */
  void setActivity(Float activity) {_activityBuffer = activity;}
  Condition condition(Condition);
  void warning(const string&);

public:

Reactor();

  // pointer to inherited will not be deleted 
  // correctly without virtual destructor
  virtual ~Reactor() { }	

  const string fqpn() const;

  /*!
    This static method is used to dynamically instantiate various
    classes of Reactor.  ReactorMaker class make an associated list of
    classname and address of instance() method of each subclass of 
    Reactor.  Each subclass of Reactor MUST have this static method.
    (Note that this must be static because only static method have
    a static address at compile time. However, note also that the compiler
    cannot check if all subclasses have this because this is
    a mere static method, not pure virtual.)
    \return a new instance.
   */
  //  static Reactor* instance() {return NULL;}
  
  virtual void initialize();
  virtual void react() = 0;
  virtual void transit() {_activity = _activityBuffer;}
  Condition status() const {return _condition;}
  void resetCondition() {_condition = Good;}

   /*!
    Returns activity of this reactor in 
    [number of molecule that this yields] / [s].
    This does not simply returns a value given in setActivity() which
    takes number of molecule per deltaT. The value given in setActivity()
    is recalculated as per second by dividing it by deltaT.
    
    \return [the number of molecule that this yield] / [s].
    \sa setActivity()
   */
  virtual Float activity();

  virtual char* description() const=0;

  void addSubstrate(Substance& substrate,int coefficient);
  void addProduct(Substance& product,int coefficient);
  void addCatalyst(Substance& catalyst,int coefficient);
  void addEffector(Substance& effector,int coefficient);

  /*!
    Returns a pointer to Reactant for ith substrate.
    This does range check so this is not the most efficient way
    to access substrates. In case access from inside of Reactor
    and efficiency matters, use vector<Reactant*> _substrateList directly.

    NOTE!!: As a matter of fact, current implementation of this
            function uses operator[] which doesn't do range check
            because vector class of SGI's stl comes with egcs/gcc
            doesn't support at() method yet.  This will be
            changed to use at() in future version.

    \return pointer to Reactant of a substrate.
    \sa Reactant
   */
  Reactant* substrate(int i) {return _substrateList[i];}
  /*!
    Returns a pointer to Reactant for ith substrate.

    \return pointer to Reactant of a substrate.
    \sa substrate
   */
  Reactant* product(int i) {return _productList[i];}
  /*!
    Returns a pointer to Reactant for ith catalyst.

    \return pointer to Reactant of a catalyst.
    \sa substrate
   */
  Reactant* catalyst(int i=0) {return _catalystList[i];}
  /*!
    Returns a pointer to Reactant for ith effector.

    \return pointer to Reactant of a effector.
    \sa substrate
   */
  Reactant* effector(int i=0) {return _effectorList[i];}

  /*!
    \return the number of substrates.
   */
  int numSubstrate() {return _substrateList.size();}
  /*!
    \return the number of products.
   */
  int numProduct()   {return _productList.size();}
  /*!
    \return the number of catalysts.
   */
  int numCatalyst()  {return _catalystList.size();}
  /*!
    \return the number of effectors.
   */
  int numEffector()  {return _effectorList.size();}



  virtual int minSubstrate() const {return 0;}
  virtual int minProduct() const {return 0;}
  virtual int minCatalyst() const {return 0;}
  virtual int minEffector() const {return 0;}

#ifdef HAVE_NUMERIC_LIMITS
  virtual int maxSubstrate() const {return numeric_limits<int>::max();}
  virtual int maxProduct() const {return numeric_limits<int>::max();}
  virtual int maxCatalyst() const {return numeric_limits<int>::max();}
  virtual int maxEffector() const {return numeric_limits<int>::max();}
#else /* HAVE_NUMERIC_LIMITS */
  virtual int maxSubstrate() const {return INT_MAX;}
  virtual int maxProduct() const {return INT_MAX;}
  virtual int maxCatalyst() const {return INT_MAX;}
  virtual int maxEffector() const {return INT_MAX;}
#endif /* HAVE_NUMERIC_LIMITS */

  static Condition globalCondition() {return _globalCondition;}

};

/*! 
  A function type that returns a pointer to Reactor.
  Mainly used to provide a way to instantiate Reactors via
  traditional C function specifically by ReactorMaker.
  Every Reactor that instantiated by ReactorMaker must have
  a this type of function which returns a instance of that Reactor.
*/
typedef Reactor* (*ReactorAllocatorFunc)();
typedef Reactor* ReactorPtr;



#endif /* ___REACTOR_H___ */
