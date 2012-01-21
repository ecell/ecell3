//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
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

#ifndef __DISCRETEEVENTSTEPPER_HPP
#define __DISCRETEEVENTSTEPPER_HPP

#include "libecs/Defs.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/Process.hpp"

#include "libecs/EventScheduler.hpp"
#include "libecs/ProcessEvent.hpp"

namespace libecs
{

LIBECS_DM_CLASS( DiscreteEventStepper, Stepper )
{
protected:
    typedef EventScheduler<ProcessEvent> ProcessEventScheduler;
    typedef ProcessEventScheduler::EventID EventID;

public:
    LIBECS_DM_OBJECT( DiscreteEventStepper, Stepper )
    {
        INHERIT_PROPERTIES( Stepper );

        PROPERTYSLOT_SET_GET( Real, Tolerance );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, TimeScale );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( String, LastProcess );
    }

    DiscreteEventStepper();
    virtual ~DiscreteEventStepper() {}

    virtual void initialize();
    virtual void step();
    virtual void interrupt( Time aTime );
    virtual void log();


    SET_METHOD( Real, Tolerance )
    {
        theTolerance = value;
    }
    
    GET_METHOD( Real, Tolerance )
    {
        return theTolerance;
    }
    
    virtual GET_METHOD( Real, TimeScale )
    {
        return 0.0;
    }

    GET_METHOD( String, LastProcess );

    ProcessVector const& getProcessVector() const
    {
        return theProcessVector;
    }

protected:
    ProcessEventScheduler    theScheduler;
    Real                        theTolerance;
    EventID                 theLastEventID;
};

} // namespace libecs

#endif /* __STEPPER_HPP */
