//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  // FIXME: contain instances, not pointers
  DECLARE_ASSOCVECTOR( String, Reactant, std::less< const String >, 
		       ReactantMap  );

  /**
     Reactor class is used to represent chemical and other phenonema which 
     may or may not result in change in quantity of one or more Substances.
  */
  class Reactor 
    : 
    public Entity
  {

  public:

    /**
       An predicate object which returns true if given pointer of Reactor 
       points to a regular Reactor (i.e. not posterior Reactor).
  
       ID of Posterior Reactors must start with '!'.
    */
    class isRegularReactor : public std::unary_function< ReactorCptr, bool >
    {
    public:
      static bool isRegularName( StringCref anID )
      {
	if( anID[0] != '!' )
	  {
	    return true;
	  }
	return false;
      }
      bool operator() ( ReactorCptr aReactorPtr ) const
      {
	return isRegularName( aReactorPtr->getID() );
      }
    };

  public: 

    // Property slots

    void setReactant( UVariableVectorRCPtrCref aMessage );

    const UVariableVectorRCPtr getReactantList() const;


  public:

    Reactor();
    virtual ~Reactor();

    virtual const PrimitiveType getPrimitiveType() const
    {
      return PrimitiveType( PrimitiveType::REACTOR );
    }

    virtual void initialize();

    virtual void differentiate() { }

    //    virtual void integrate() { }

    virtual void compute() { }

    /**
       Set activity variable.  This must be set at every turn and takes
       [number of molecule that this reactor yields] / [deltaT].
       However, public activity() method returns it as number of molecule
       per a second, not per deltaT.

       @param activity [number of molecule that this yields] / [deltaT].
       @see getActivity(), getActivityPerSecond()
    */
    void setActivity( RealCref anActivity ) 
    { 
      theActivity = anActivity; 
    }

    /**
       Returns activity of this reactor in 
       [number of molecule that this yields] / [s].
       This does not simply returns a value given in setActivity() which
       takes number of molecule per deltaT. The value given in setActivity()
       is recalculated as per second by dividing it by deltaT.
    
       @return [the number of molecule that this yield] / [s].
       @see setActivity()
    */

    virtual const Real getActivity() const
    {
      return theActivity;
    }

    const Real getInitialActivity() const
    {
      return theInitialActivity;
    }

    void setInitialActivity( RealCref anActivity );

    void registerReactant( StringCref aName, FullIDCref aFullID,
			   const Int aStoichiometry );

    void registerReactant( StringCref aName, SubstancePtr aSubstance, 
			   const Int aStoichiometry );

    /**
       Get Reactant by tag name.

       @param aReactantName
       @return a Reactant
       @see Reactant
    */

    Reactant getReactant( StringCref aReactantName );


    /**
       @return the number of substrates.
    */
    const Int getNumberOfReactants() const;

  protected:

    // convenience inline methods for use in subclasses

    PropertySlotPtr getPropertySlotOfReactant( StringCref aReactantName,
					       StringCref aPropertyName );

    void makeSlots();

  protected:

    ReactantMap theReactantMap;

    Real        theActivity;

  private:

    Real theInitialActivity;

  };

  /** 
      A function type that returns a pointer to Reactor.
      Mainly used to provide a way to instantiate Reactors via
      traditional C function specifically by ReactorMaker.
      Every Reactor that instantiated by ReactorMaker must have
      a this type of function which returns a instance of that Reactor.
  */
  typedef ReactorPtr (*ReactorAllocatorFunc)();

  /** @} */ //end of libecs_module \

} // namespace libecs

#endif /* ___REACTOR_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
