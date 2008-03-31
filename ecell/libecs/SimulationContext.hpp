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

#ifndef __SIMULATIONCONTEXT_HPP
#define __SIMULATIONCONTEXT_HPP

#include "libecs.hpp"
#include "SystemStepper.hpp"
#include "EventScheduler.hpp"
#include "StepperEvent.hpp"

#ifdef DLL_EXPORT
#undef DLL_EXPORT
#include <gsl/gsl_rng.h>
#define DLL_EXPORT
#else
#include <gsl/gsl_rng.h>
#endif

namespace libecs {

class SimulationContext
{
public:
    typedef EventScheduler<StepperEvent> StepperEventScheduler;
    typedef StepperEventScheduler::EventIndex EventIndex;

public:
    SimulationContext();

    ~SimulationContext();

    void setRngSeed( const String& value );

    Model* getModel() {
        return model_;
    }

    const Model* getModel() const
    {
        return model_;
    }

    void setModel( Model* model );

    void startup();

    void initialize();

    /**
       Get the next event to occur on the scheduler.
     */
    const StepperEvent& getTopEvent() const
    {
        return scheduler_.getTopEvent();
    }

    const SystemStepper& getSystemStepper() const
    {
        return systemStepper_;
    }

    SystemStepper& getSystemStepper()
    {
        return systemStepper_;
    }

    /**
       Get the last event executed by the scheduler.

     */
    const StepperEvent& getLastEvent() const
    {
        return lastEvent_;
    }

    const StepperEventScheduler&  getScheduler() const
    {
        return scheduler_;
    }

    StepperEventScheduler&  getScheduler()
    {
        return scheduler_;
    }

    const gsl_rng* getRng() const
    {
        return rng_;
    }

    System* getWorld()
    {
        return &world_;
    }

protected:
    void initializeSteppers();
    void postInitializeSteppers();
    void ensureExistenceOfSizeVariable();

private:
    Model*                model_;
    SystemStepper         systemStepper_;
    System                world_;
    StepperEvent          lastEvent_;
    StepperEventScheduler scheduler_;
    gsl_rng*              rng_;
};

} // namespace libecs

#endif /* __SIMULATIONCONTEXT_HPP */
