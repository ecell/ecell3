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

#ifndef __VARIABLEREFERENCE_HPP
#define __VARIABLEREFERENCE_HPP

#include "libecs.hpp"
#include "Variable.hpp"

namespace libecs
{

  /** @addtogroup entities
   *@{
   */


  /** @file */

  class VariableReference
  {

  public:

    class CoefficientCompare
    {
    public:

      CoefficientCompare()
      {
	; // do nothing
      }

      bool operator()( VariableReferenceCref aLhs, 
		       VariableReferenceCref aRhs ) const
      {
	return compare( aLhs.getCoefficient(), aRhs.getCoefficient() );
      }

      bool operator()( const Integer aLhs, 
		       VariableReferenceCref aRhs ) const
      {
	return compare( aLhs, aRhs.getCoefficient() );
      }

      bool operator()( VariableReferenceCref aLhs, 
		       const Integer aRhs ) const
      {
	return compare( aLhs.getCoefficient(), aRhs );
      }


    private:

      static const std::less<Integer> compare;

    };

    class NameCompare
    {
    public:

      NameCompare()
      {
	; // do nothing
      }

      bool operator()( VariableReferenceCref aLhs, 
		       VariableReferenceCref aRhs ) const
      {
	return compare( aLhs.getName(), aRhs.getName() );
      }

      bool operator()( StringCref aLhs, 
		       VariableReferenceCref aRhs ) const
      {
	return compare( aLhs, aRhs.getName() );
      }

      bool operator()( VariableReferenceCref aLhs, 
		       StringCref aRhs ) const
      {
	return compare( aLhs.getName(), aRhs );
      }


    private:

      static const std::less<String> compare;

    };

  public:

    VariableReference()
      :
      theVariablePtr( NULLPTR ),
      theCoefficient( 0 ),
      theIsAccessor( true )
    {
      ; // do nothing
    }

    VariableReference( StringCref aName, 
		       VariablePtr aVariablePtr, 
		       const Integer aCoefficient,
		       const bool anIsAccessor = true )  
      : 
      theName( aName ),
      theVariablePtr( aVariablePtr ), 
      theCoefficient( aCoefficient ),
      theIsAccessor( anIsAccessor )
    {
      ; // do nothing
    }

    ~VariableReference() {}

    void setName( StringCref aName )
    {
      theName = aName;
    }


    // can there be unnamed VariableReferences?
    const String getName() const 
    { 
      return theName; 
    }

    void setVariable( VariablePtr const aVariablePtr )
    {
      theVariablePtr = aVariablePtr;
    }

    const VariablePtr getVariable() const 
    { 
      return theVariablePtr; 
    }

    void setCoefficient( IntegerCref aCoefficient )
    {
      theCoefficient = aCoefficient;
    }

    const Integer getCoefficient() const 
    { 
      return theCoefficient; 
    }

    const bool isMutator() const
    {
      if( theCoefficient == 0 )
	{
	  return false;
	}
      else
	{
	  return true;
	}
    }

    void setIsAccessor( const bool anIsAccessor )
    {
      theIsAccessor = anIsAccessor;
    }

    const bool isAccessor() const
    {
      return theIsAccessor;
    }

    const Real getValue() const
    {
      return theVariablePtr->getValue();
    }

    void setValue( const Real aValue ) const
    {
      theVariablePtr->setValue( aValue );
    }

    /**
       Add a value to the variable according to the coefficient.
       
       Set a new value to the variable.  
       The new value is: old_value + ( aValue * theCoeffiencnt ).

       @param aValue a Real value to be added.
    */

    void addValue( const Real aValue ) const
    {
      theVariablePtr->addValue( aValue );
    }

    const Real getMolarConc() const
    {
      return theVariablePtr->getMolarConc();
    }

    const Real getNumberConc() const
    {
      return theVariablePtr->getNumberConc();
    }

    const Real getTotalVelocity() const
    {
      return theVariablePtr->getTotalVelocity();
    }

    // be careful.. you may mean getTotalVelocity(), not getVelocity()
    const Real getVelocity() const
    {
      return theVariablePtr->getVelocity();
    }

    // before trying this consider addFlux() below.
    void addVelocity( const Real aValue ) const
    {
      theVariablePtr->addVelocity( aValue );
    }

    const bool isFixed() const
    {
      return theVariablePtr->isFixed();
    }

    void setFixed( const bool aValue ) const
    {
      theVariablePtr->setFixed( aValue );
    }


    void addFlux( const Real aVelocity ) const
    {
      theVariablePtr->addVelocity( aVelocity * theCoefficient );
    }

    SystemPtr getSuperSystem() const
    {
      return theVariablePtr->getSuperSystem();
    }


  private:

    String      theName;
    VariablePtr theVariablePtr;
    Integer     theCoefficient;
    bool        theIsAccessor;

  };

  //@}

} // namespace libecs


#endif /* __VARIABLEREFERENCE_HPP */

