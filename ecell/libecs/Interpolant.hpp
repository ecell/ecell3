//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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

#ifndef __INTERPOLANT_HPP
#define __INTERPOLANT_HPP

#include "libecs.hpp"

namespace libecs
{

class Stepper;
class Variable;

class LIBECS_API Interpolant
{
public:

    class VariablePtrComparator
    {
    public:
        bool operator()( const Interpolant* const aLhs,
                         const Interpolant* const aRhs ) const
        {
            return compare( aLhs->getVariable(), aRhs->getVariable() );
        }

        bool operator()( const Interpolant* const aLhs,
                         const Variable* const aRhs ) const
        {
            return compare( aLhs->getVariable(), aRhs );
        }

        bool operator()( const Variable* const aLhs,
                         const Interpolant* const aRhs ) const
        {
            return compare( aLhs, aRhs->getVariable() );
        }

    private:
        // if statement can be faster than returning an expression directly
        inline static bool compare( const Variable* const aLhs,
                                    const Variable* const aRhs )
        {
            return aLhs < aRhs;
        }
    };


    Interpolant() {}

    virtual void setVariable( Variable *var )
    {
        theVariable = var;
    }

    virtual ~Interpolant() {}

    virtual
    const Real getVelocity( RealParam aTime ) const
    {
        return 0.0;
    }

    virtual
    const Real getDifference( RealParam aTime, RealParam anInterval ) const
    {
        return 0.0;
    }

    Variable* const getVariable() const
    {
        return theVariable;
    }

private:
    Variable* theVariable;

};

}


#endif /* __INTERPOLANT_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
