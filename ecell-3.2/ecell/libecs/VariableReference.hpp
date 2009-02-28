//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

class LIBECS_API VariableReference
{

public:
    class CoefficientLess
    {
    public:
        CoefficientLess()
        {
            ; // do nothing
        }

        bool operator()( VariableReferenceCref aLhs, 
                         VariableReferenceCref aRhs ) const
        {
            return compare( aLhs.getCoefficient(), aRhs.getCoefficient() );
        }

        bool operator()( IntegerParam aLhs, 
                         VariableReferenceCref aRhs ) const
        {
            return compare( aLhs, aRhs.getCoefficient() );
        }

        bool operator()( VariableReferenceCref aLhs, 
                         IntegerParam aRhs ) const
        {
            return compare( aLhs.getCoefficient(), aRhs );
        }

    private:
        static const bool compare( IntegerParam aLhs, IntegerParam aRhs )
        {
            return std::less<Integer>()( aLhs, aRhs );
        }
    };

    class NameLess
    {
    public:
        NameLess()
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

        static const bool compare( StringCref aLhs, StringCref aRhs )
        {
            const bool anIsLhsEllipsis(
                VariableReference::isEllipsisNameString( aLhs ) );
            const bool anIsRhsEllipsis(
                VariableReference::isEllipsisNameString( aRhs ) );

            // both are ellipses, or both are normal names.
            if( anIsLhsEllipsis == anIsLhsEllipsis )
            {
                return std::less<String>()( aLhs, aRhs );
            }
            else // always sort ellipses last
            {
                return anIsRhsEllipsis;
            }
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

        bool operator()( VariableReferenceCref aLhs, 
                         VariableReferenceCref aRhs ) const
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
                return NameLess()( aLhs, aRhs );
            }
        }
    };

public:
    VariableReference()
        : theVariable( NULLPTR ),
          theCoefficient( 0 ),
          theIsAccessor( true )
    {
        ; // do nothing
    }

    VariableReference( StringCref aName, 
                       Variable* aVariable, 
                       IntegerParam aCoefficient,
                       const bool anIsAccessor = true )    
        : theName( aName ),
          theVariable( aVariable ), 
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

    void setVariable( Variable* aVariable )
    {
        theVariable = aVariable;
    }

    Variable* getVariable() const 
    { 
        return theVariable; 
    }

    void setCoefficient( IntegerParam aCoefficient )
    {
        theCoefficient = aCoefficient;
    }

    const Integer getCoefficient() const 
    { 
        return theCoefficient; 
    }

    const bool isMutator() const
    {
        return theCoefficient != 0;
    }

    void setIsAccessor( const bool anIsAccessor )
    {
        theIsAccessor = anIsAccessor;
    }

    const bool isAccessor() const
    {
        return theIsAccessor;
    }

    void setValue( RealParam aValue ) const
    {
        theVariable->setValue( aValue );
    }

    const Real getValue() const
    {
        return theVariable->getValue();
    }

    /**
       Add a value to the variable according to the coefficient.
       
       Set a new value to the variable.    
       The new value is: old_value + ( aValue * theCoeffiencnt ).

       @param aValue a Real value to be added.
    */
    void addValue( RealParam aValue ) const
    {
        theVariable->addValue( aValue * theCoefficient );
    }

    const Real getMolarConc() const
    {
        return theVariable->getMolarConc();
    }

    const Real getNumberConc() const
    {
        return theVariable->getNumberConc();
    }

    const Real getVelocity() const
    {
        return theVariable->getVelocity();
    }

    const bool isFixed() const
    {
        return theVariable->isFixed();
    }

    void setFixed( const bool aValue ) const
    {
        theVariable->setFixed( aValue );
    }

    SystemPtr getSuperSystem() const
    {
        return theVariable->getSuperSystem();
    }

    const bool isEllipsisName() const
    {
        return isEllipsisNameString( theName );
    }

    const Integer getEllipsisNumber() const;

    const bool isDefaultName() const
    {
        return isDefaultNameString( theName );
    }

    bool operator==( VariableReferenceCref rhs ) const
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

    static const bool isEllipsisNameString( StringCref aName )
    {
        return aName.size() > 3 && aName.compare( 0, 3, ELLIPSIS_PREFIX ) == 0
               && std::isdigit( *reinterpret_cast< const unsigned char* >(
                    &aName[ 4 ] ) );
    }

    static const bool isDefaultNameString( StringCref aName )
    {
        return aName == DEFAULT_NAME;
    }

public:
    static const String ELLIPSIS_PREFIX;
    static const String DEFAULT_NAME;

private:
    String            theName;
    Variable*         theVariable;
    Integer           theCoefficient;
    bool              theIsAccessor;
};

} // namespace libecs

#endif /* __VARIABLEREFERENCE_HPP */

