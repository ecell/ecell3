//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

#ifndef ___REACTOR_H___
#define ___REACTOR_H___

// #include <climits>
#include <vector>

#include "libecs.hpp"
#include "Entity.hpp"

namespace libecs
{

  // FIXME: contain instances, not pointers
  DECLARE_VECTOR( ReactantPtr, ReactantVector );

  /**
     Reactor class is used to represent chemical and other phenonema which 
     may or may not result in change in quantity of one or more Substances.
  */
  class Reactor : public Entity
  {

  public:

    /**
       An predicate object which returns true if given pointer of Reactor 
       points to a regular Reactor (i.e. not posterior Reactor).
  
       ID of Posterior Reactors must start with '!'.
    */
    class isRegularReactor : public unary_function< ReactorCptr, bool >
    {
    public:
      static bool isRegularName( StringCref id )
      {
	if( id[0] != '!' )
	  {
	    return true;
	  }
	return false;
      }
      bool operator() ( ReactorCptr r ) const
      {
	return isRegularName( r->getID() );
      }
    };

  public: 

    // PropertyInterfaces

    void setAppendSubstrate( MessageCref message );
    void setAppendProduct( MessageCref message );
    void setAppendCatalyst( MessageCref message );
    void setAppendEffector( MessageCref message );

    void setSubstrateList( MessageCref message );
    void setProductList( MessageCref message );
    void setCatalystList( MessageCref message );
    void setEffectorList( MessageCref message );

    void setInitialActivity( MessageCref message );

    const Message getSubstrateList( StringCref keyword );
    const Message getProductList( StringCref keyword );
    const Message getCatalystList( StringCref keyword );
    const Message getEffectorList( StringCref keyword );

    const Message getInitialActivity( StringCref keyword );


    void appendSubstrate( FullIDCref fullid, int coefficient );
    void appendProduct  ( FullIDCref fullid, int coefficient );
    void appendCatalyst ( FullIDCref fullid, int coefficient );
    void appendEffector ( FullIDCref fullid, int coefficient );
    void setInitialActivity( RealCref activity );

    /**
       Reactor Condition enum is used by Reactor's self-diagnosis system.

       There are two types of Reactor conditions, local and global.
       Local Reactor condition accessed by status() method indicates
       condition of the Reactor.
       Global Reactor condition can be checked by globalCondition() method
       and it returns Bad if there is at least one Reactor in the RootSystem
       whose status is not Good.

       @see status() globalCondition()
    */
    enum Condition {
      /// Condition Good. Everything seems to be Ok. 
      Good = 0x00,
      /// Something wrong was occured at the time of initialization.
      InitFail = 0x01,
      /// Failed to do something at the time of react phase. 
      ReactFail = 0x02,
      /// Something is wrong, but don't know what is it.
      Bad = 0x04,
      /// Fatal.  Cannot continue. 
      Fatal = 0x08,
      /// Premature. Need initialization to work correctly. */
      Premature= 0x16
    };

    enum LigandType { Substrate, Product, Catalyst, Effector };
    static const char* LIGAND_STRING_TABLE[];

  private:

    Real theInitialActivity;
    Real theActivityBuffer;
    Condition theCondition;
    static Condition theGlobalCondition;

  protected:

    Real theActivity;

    void makeSlots();

    ReactantVector theSubstrateList; 
    ReactantVector theProductList;   
    ReactantVector theEffectorList;
    ReactantVector theCatalystList;

    /**
       Set activity variable.  This must be set at every turn and takes
       [number of molecule that this reactor yields] / [deltaT].
       However, public activity() method returns it as number of molecule
       per a second, not per deltaT.

       @param activity [number of molecule that this yields] / [deltaT].
       @see activity()
    */
    void setActivity( RealCref activity ) 
    { 
      theActivityBuffer = activity; 
    }

    Condition condition( Condition );
    void warning( StringCref );

  public:

    Reactor();
    virtual ~Reactor();

    const FullID getFullID() const;

    virtual const PrimitiveType getPrimitiveType() const
    {
      return PrimitiveType( PrimitiveType::REACTOR );
    }

    virtual void initialize();
    virtual void react() = 0;
    virtual void transit() { theActivity = theActivityBuffer; }
    Condition status() const { return theCondition; }
    void resetCondition() { theCondition = Good; }

    /**
       Returns activity of this reactor in 
       [number of molecule that this yields] / [s].
       This does not simply returns a value given in setActivity() which
       takes number of molecule per deltaT. The value given in setActivity()
       is recalculated as per second by dividing it by deltaT.
    
       @return [the number of molecule that this yield] / [s].
       @see setActivity()
    */
    virtual Real getActivity();

    void appendSubstrate( SubstanceRef substrate, int coefficient );
    void appendProduct( SubstanceRef product, int coefficient );
    void appendCatalyst( SubstanceRef catalyst, int coefficient );
    void appendEffector( SubstanceRef effector, int coefficient );

    /**
       Returns a pointer to a Reactant of ith substrate.
       FIXME: range check?

       @return pointer to a Reactant of the substrate.
       @see Reactant
    */
    ReactantPtr getSubstrate( int i = 0 ) { return theSubstrateList[i]; }

    /**
       Returns a pointer to a Reactant of ith substrate.

       @return pointer to a Reactant of the substrate.
       @see substrate
    */
    ReactantPtr getProduct( int i = 0 ) { return theProductList[i]; }

    /**
       Returns a pointer to Reactant of ith catalyst.

       @return pointer to Reactant of the catalyst.
       @see substrate
    */
    ReactantPtr getCatalyst( int i = 0 ) { return theCatalystList[i]; }

    /**
       Returns a pointer to Reactant for ith effector.

       @return pointer to Reactant of a effector.
       @see substrate
    */
    ReactantPtr getEffector( int i = 0 ) { return theEffectorList[i]; }

    /**
       @return the number of substrates.
    */
    int getNumberOfSubstrates() { return theSubstrateList.size(); }

    /**
       @return the number of products.
    */
    int getNumberOfProducts()   { return theProductList.size(); }

    /**
       @return the number of catalysts.
    */

    int getNumberOfCatalysts()  { return theCatalystList.size(); }

    /**
       @return the number of effectors.
    */
    int getNumberOfEffectors()  { return theEffectorList.size(); }

    virtual int getMinimumNumberOfSubstrates() const { return 0; }
    virtual int getMinimumNumberOfProducts() const { return 0; }
    virtual int getMinimumNumberOfCatalysts() const { return 0; }
    virtual int getMinimumNumberOfEffectors() const { return 0; }

#ifdef HAVE_NUMERIC_LIMITS
    virtual int getMaximumNumberOfSubstrates() const 
    { return numeric_limits<int>::max(); }
    virtual int getMaximumNumberOfProducts() const 
    { return numeric_limits<int>::max(); }
    virtual int getMaximumNumberOfCatalysts() const 
    { return numeric_limits<int>::max(); }
    virtual int getMaximumNumberOfEffectors() const 
    { return numeric_limits<int>::max(); }
#else /* HAVE_NUMERIC_LIMITS */
    virtual int getMaximumNumberOfSubstrates() const {return INT_MAX;}
    virtual int getMaximumNumberOfProducts() const {return INT_MAX;}
    virtual int getMaximumNumberOfCatalysts() const {return INT_MAX;}
    virtual int getMaximumNumberOfEffectors() const {return INT_MAX;}
#endif /* HAVE_NUMERIC_LIMITS */

    static Condition getGlobalCondition() {return theGlobalCondition;}

  };

  /** 
      A function type that returns a pointer to Reactor.
      Mainly used to provide a way to instantiate Reactors via
      traditional C function specifically by ReactorMaker.
      Every Reactor that instantiated by ReactorMaker must have
      a this type of function which returns a instance of that Reactor.
  */
  typedef ReactorPtr (*ReactorAllocatorFunc)();


} // namespace libecs

#endif /* ___REACTOR_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
