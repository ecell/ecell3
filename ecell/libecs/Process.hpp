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

#ifndef ___PROCESS_H___
#define ___PROCESS_H___

#include "AssocVector.h"

#include "libecs.hpp"
#include "Entity.hpp"
#include "VariableReference.hpp"

namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


  DECLARE_ASSOCVECTOR( String, VariableReference, std::less< const String >, 
		       VariableReferenceMap  );

  DECLARE_VECTOR( VariableReference, VariableReferenceVector );

  /**
     Process class is used to represent chemical and other phenonema which 
     may or may not result in change in value of one or more Variables.

  */

  class Process 
    : 
    public Entity
  {

  public: 


    class PriorityCompare
    {
    public:
      bool operator()( ProcessPtr aLhs, ProcessPtr aRhs ) const
      {
	return compare( aLhs->getPriority(), aRhs->getPriority() );
      }

      bool operator()( ProcessPtr aLhs, const Int aRhs ) const
      {
	return compare( aLhs->getPriority(), aRhs );
      }

      bool operator()( const Int aLhs, ProcessPtr aRhs ) const
      {
	return compare( aLhs, aRhs->getPriority() );
      }

    private:

      // if statement can be faster than returning an expression directly
      inline static bool compare( const Int aLhs, const Int aRhs )
      {
	if( aLhs < aRhs )
	  {
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }


    };

    /** 
	A function type that returns a pointer to Process.  

	Every Process class must have this type of a function which returns
	an instance for the ProcessMaker.
    */

    typedef ProcessPtr (* AllocatorFuncPtr )();


  public:

    Process();
    virtual ~Process();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::PROCESS );
    }

    virtual void initialize();
    
    virtual void process() = 0;
    
    
    /**
       Set activity value.

       Semantics of this property can be defined in each subclass of
       Process.  Usually it is a turnover number if the Process represents a
       chemical reaction.

       If the value has time in its dimension, the unit should be [per
       second], not [per step], to keep its meaning in
       multi-stepper simulations.

       @param anActivity An activity value to be set.
       @see getActivity()
    */

    void setActivity( RealCref anActivity ) 
    { 
      theActivity = anActivity; 
    }

    /**
       Get activity value.

       @see setActivity()
       @return the activity value of this Process.
    */

    const Real getActivity() const
    {
      return theActivity;
    }

    void setVariableReference( PolymorphVectorCref aValue );

    void setVariableReferenceList( PolymorphCref );

    const Polymorph getVariableReferenceList() const;

    void removeVariableReference( StringCref aName );

    void registerVariableReference( StringCref aName, FullIDCref aFullID,
				    const Int aCoefficient );

    void registerVariableReference( StringCref aName, VariablePtr aVariable, 
				    const Int aCoefficient );

    /**
       Get VariableReference by tag name.

       @param aVariableReferenceName
       @return a VariableReference
       @see VariableReference
    */

    VariableReference getVariableReference( StringCref aVariableReferenceName );

    VariableReferenceVectorIterator findVariableReference( StringCref aName );

    /**
       @return a const reference to the VariableReferenceVector
    */

    VariableReferenceVectorCref getVariableReferenceVector() const
    {
      return theVariableReferenceVector;
    }

    void setPriority( IntCref aValue )
    {
      thePriority = aValue;
    }

    const Int getPriority() const
    {
      return thePriority;
    }

  protected:

    void makeSlots();

  protected:

    VariableReferenceVector theVariableReferenceVector;

    VariableReferenceVectorConstIterator theFirstZeroVariableReferenceIterator;
    VariableReferenceVectorConstIterator 
    theFirstPositiveVariableReferenceIterator;

  private:

    Real        theActivity;
    Int         thePriority;

  };





  /*@}*/

} // namespace libecs

#endif /* ___PROCESS_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
