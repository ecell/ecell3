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

#include "AssocVector.h"

#include "libecs.hpp"
#include "Entity.hpp"
#include "Reactant.hpp"

namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


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
	A function type that returns a pointer to Reactor.  

	Every Reactor class must have this type of a function which returns
	an instance for the ReactorMaker.
    */

    typedef ReactorPtr (* AllocatorFuncPtr )();


  public:

    Reactor();
    virtual ~Reactor();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::REACTOR );
    }

    virtual void initialize();
    
    virtual void react() = 0;
    
    
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

    const Real getActivity() const
    {
      return theActivity;
    }

    /**
       Returns activity value (per second).

       Default action of this method is to return getActivity() / step
       interval, but this action can be changed in subclasses.

       @return activity of this Entity per second
    */

    const Real getActivityPerSecond() const;



    void setReactant( PolymorphCref aValue );

    const Polymorph getReactantList() const;

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
       @return a const reference to the reactant map
    */
    ReactantMapCref getReactantMap() const
    {
      return theReactantMap;
    }

  protected:

    void makeSlots();

  protected:

    ReactantMap theReactantMap;


  private:

    Real        theActivity;

  };





  /*@}*/

} // namespace libecs

#endif /* ___REACTOR_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
