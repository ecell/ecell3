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
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
// E-Cell Project.
//

#ifndef __LOGGING_POLICY_HPP
#define __LOGGING_POLICY_HPP

#include "libecs.hpp"
#include "Exceptions.hpp"

namespace libecs {

class LoggingPolicy
{
public:
    // FIXME: should be referred to by VVector.
    enum EndPolicy
    {
        ABORT_ON_FULL = 0,
        OVERWRITE = 1
    };
public:
    LoggingPolicy( IntegerParam minimumStep,
            RealParam    minimumInterval,
            EndPolicy endPolicy,
            IntegerParam maxSpace )
        : minimumStep_( minimumStep ),
          minimumInterval_( minimumInterval ),
          endPolicy_( endPolicy ),
          maxSpace_( maxSpace )
    {
        if ( minimumStep < 0 || minimumInterval < 0 )
        {
            THROW_EXCEPTION( ValueError,
                             "the minimum step and the minimum time interval "
                             "must be positive numbers" );
        }
    }

    LoggingPolicy()
        : minimumStep_( 1 ),
          minimumInterval_( 0.0 ),
          endPolicy_( ABORT_ON_FULL ),
          maxSpace_( 0 )
    {
    }
        
          
    Integer getMinimumStep() const
    {
        return minimumStep_;
    }

    Real getMinimumInterval() const
    {
        return minimumInterval_;
    }

    EndPolicy getEndPolicy() const
    {
        return endPolicy_;
    }

    Integer getMaxSpace() const
    {
        return maxSpace_;
    }

private:
    Integer minimumStep_;
    Real    minimumInterval_;
    EndPolicy endPolicy_;
    Integer maxSpace_;
};

} // namespace libecs


#endif /* __LOGGING_POLICY_HPP */
