//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __VARIABLEREFERENCE_HPP
#define __VARIABLEREFERENCE_HPP

#include "libecs/Defs.hpp"
#include "libecs/Variable.hpp"
#include <cctype>

namespace libecs
{

class Process;

class LIBECS_API VariableReference
{
    friend class Process;
public:
    class CoefficientLess
    {
    public:
        CoefficientLess()
        {
            ; // do nothing
        }

        bool operator()( VariableReference const& aLhs, 
                         VariableReference const& aRhs ) const
        {
            return compare( aLhs.getCoefficient(), aRhs.getCoefficient() );
        }

        bool operator()( Integer aLhs, 
                         VariableReference const& aRhs ) const
        {
            return compare( aLhs, aRhs.getCoefficient() );
        }

        bool operator()( VariableReference const& aLhs, 
                         Integer aRhs ) const
        {
            return compare( aLhs.getCoefficient(), aRhs );
        }

    private:
        static bool compare( Integer aLhs, Integer aRhs )
        {
            return std::less<Integer>()( aLhs, aRhs );
        }
    };

    class FullIDLess 
    {
    public:
        FullIDLess()
        {
            ; // do nothing
        }

        bool operator()( VariableReference const& aLhs, 
                         VariableReference const& aRhs ) const
        {
            return compare( aLhs.getFullID(), aRhs.getFullID() );
        }

    private:

        static bool compare( FullID const& aLhs, FullID const& aRhs )
        {
            return std::less<String>()( aLhs, aRhs );
        }
    };


    // compare coefficients first, and if equal, compare names.
    class Less
    {
    public:
        Less()
        {
            ; // do nothing
        }

        bool operator()( VariableReference const& aLhs, 
                         VariableReference const& aRhs ) const
        {
            CoefficientLess aCoefficientLess;
            if( aCoefficientLess( aLhs, aRhs ) )
            {
                return true;
            }
            else if( aCoefficientLess( aRhs, aLhs ) )
            {
                return false;
            } 
            else // lhs.coeff == rhs.coeff
            {
                return FullIDLess()( aLhs, aRhs );
            }
        }
    };

public:
    VariableReference()
        : theFullID(),
          theVariable( 0 ),
          theCoefficient( 0 ),
          theIsAccessor( true )
    {
        ; // do nothing
    }

    VariableReference( Integer aSerial,
                       FullID const& anFullID,
                       Integer aCoefficient,
                       const bool anIsAccessor = true )    
        : theSerial( aSerial ),
          theFullID( anFullID ),
          theVariable( 0 ),
          theCoefficient( aCoefficient ),
          theIsAccessor( anIsAccessor )
    {
        ; // do nothing
    }

    VariableReference( Integer aSerial,
                       Variable* aVariable,
                       Integer aCoefficient,
                       const bool anIsAccessor = true )    
        : theSerial( aSerial ),
          theFullID(),
          theVariable( aVariable ),
          theCoefficient( aCoefficient ),
          theIsAccessor( anIsAccessor )
    {
        ; // do nothing
    }

    ~VariableReference() {}

    void setName( String const& aName )
    {
        theName = aName;
    }

    const String getName() const 
    { 
        return theName; 
    }

    const Integer getSerial() const
    {
        return theSerial;
    }

    FullID const& getFullID() const
    {
        return theFullID;
    }

    Variable* getVariable() const 
    { 
        return theVariable; 
    }

    void setCoefficient( Integer aCoefficient )
    {
        theCoefficient = aCoefficient;
    }

    Integer getCoefficient() const 
    { 
        return theCoefficient; 
    }

    bool isMutator() const
    {
        return theCoefficient != 0;
    }

    void setIsAccessor( const bool anIsAccessor )
    {
        theIsAccessor = anIsAccessor;
    }

    bool isAccessor() const
    {
        return theIsAccessor;
    }

    bool isEllipsisName() const
    {
        return isEllipsisNameString( theName );
    }

    Integer getEllipsisNumber() const;

    bool isDefaultName() const
    {
        return isDefaultNameString( theName );
    }

    bool operator==( VariableReference const& rhs ) const
    {
        if( theName        == rhs.theName && 
            theVariable == rhs.theVariable &&
            theCoefficient == rhs.theCoefficient &&
            theIsAccessor  == rhs.theIsAccessor )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    static bool isEllipsisNameString( String const& aName )
    {
        return aName.size() > 3 && aName.compare( 0, 3, ELLIPSIS_PREFIX ) == 0
               && std::isdigit( *reinterpret_cast< const unsigned char* >(
                    &aName[ 4 ] ) );
    }

    static bool isDefaultNameString( String const& aName )
    {
        return aName == DEFAULT_NAME;
    }

    LIBECS_DEPRECATED
    void setValue( Real aValue ) const
    {
        theVariable->setValue( aValue );
    }

    LIBECS_DEPRECATED
    Real getValue() const
    {
        return theVariable->getValue();
    }

    /**
       Add a value to the variable according to the coefficient.
       
       Set a new value to the variable.    
       The new value is: old_value + ( aValue * theCoeffiencnt ).

       @param aValue a Real value to be added.
       @deprecated
    */
    LIBECS_DEPRECATED
    void addValue( Real aValue ) const
    {
        theVariable->addValue( aValue * theCoefficient );
    }

    LIBECS_DEPRECATED
    Real getMolarConc() const
    {
        return theVariable->getMolarConc();
    }

    LIBECS_DEPRECATED
    Real getNumberConc() const
    {
        return theVariable->getNumberConc();
    }

    LIBECS_DEPRECATED
    Real getVelocity() const
    {
        return theVariable->getVelocity();
    }

    LIBECS_DEPRECATED
    bool isFixed() const
    {
        return theVariable->isFixed();
    }

    LIBECS_DEPRECATED
    void setFixed( const bool aValue ) const
    {
        theVariable->setFixed( aValue );
    }

    LIBECS_DEPRECATED
    System* getSuperSystem() const
    {
        return theVariable->getSuperSystem();
    }

protected:

    void setSerial( Integer anID )
    {
        theSerial = anID;
    }

    void setFullID( FullID const& aFullID )
    {
        theFullID = aFullID;
    }

    void setVariable( Variable* aVariable )
    {
        theVariable = aVariable;
    }


public:
    static const String ELLIPSIS_PREFIX;
    static const String DEFAULT_NAME;

private:
    Integer           theSerial;
    String            theName;
    FullID            theFullID;
    Variable*         theVariable;
    Integer           theCoefficient;
    bool              theIsAccessor;
};

} // namespace libecs

#endif /* __VARIABLEREFERENCE_HPP */

