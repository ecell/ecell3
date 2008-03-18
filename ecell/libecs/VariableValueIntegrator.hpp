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

#ifndef __VARIABLEVALUEINTEGRATOR_HPP
#define __VARIABLEVALUEINTEGRATOR_HPP

#include "Variable.hpp"
#include "Interpolant.hpp"

namespace libecs {

class VariableValueIntegrator
{
public:
    typedef std::vector<Interpolant*> InterpolantVector;

public:
    VariableValueIntegrator( Variable* var )
        : var_( var )
    {
    }

    /**
       Register an interpolant to this variable.
     */
    void addInterpolant( Interpolant* interpolant )
    {
        interpolant->setVariable( var_ );
        interpolants_.push_back( interpolant );
    }

    void removeInterpolants()
    {
        for ( InterpolantVector::iterator i( interpolants_.begin() );
                i != interpolants_.end(); ++i )
        {
            delete *i;
        }
        interpolants_.clear();
    }

    const Real calculateDifferenceSum( TimeParam currentTime,
                                       TimeParam interval ) const
    {
        Real retval( 0.0 );
        for ( InterpolantVector::const_iterator i( interpolants_.begin() );
                i != interpolants_.end(); ++i )
        {
            retval += (*i)->getDifference( currentTime, interval );
        }

        return retval;
    }

    const Real calculateVelocitySum( TimeParam currentTime ) const
    {
        Real retval( 0.0 );
        for ( InterpolantVector::const_iterator i( interpolants_.begin() );
                i != interpolants_.end(); ++i )
        {
            retval += (*i)->getVelocity( currentTime );
        }
        return retval;
    }

    void integrate( TimeParam currentTime )
    {
        var_->setValue( var_->getValue()
                + calculateDifferenceSum( currentTime,
                    getElapsedTime( currentTime ) ) );
    }

    void update( TimeParam currentTime )
    {
        integrate( currentTime );
        lastUpdated_ = currentTime;
    }

    Time getElapsedTime( TimeParam currentTime )
    {
        return currentTime - lastUpdated_;
    }

    Time getLastUpdateTime() const
    {
        return lastUpdated_;
    }

private:
    Time lastUpdated_;
    InterpolantVector interpolants_;
    Variable* var_;
};

} // namespace libecs
#endif /* __VARIABLEVALUEINTEGRATOR_HPP */
