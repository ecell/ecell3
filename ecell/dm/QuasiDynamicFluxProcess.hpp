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

#ifndef __QUASIDYNAMICFLUXPROCESS_HPP
#define __QUASIDYNAMICFLUXPROCESS_HPP

#include <libecs/libecs.hpp>
#include <libecs/ContinuousProcess.hpp>
#include <libecs/Util.hpp>
#include <libecs/FullID.hpp>
#include <libecs/PropertyInterface.hpp>

#include <libecs/System.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/Variable.hpp>
#include <libecs/Interpolant.hpp>

#include "QuasiDynamicFluxProcessInterface.hpp"

LIBECS_DM_CLASS_EXTRA_1( QuasiDynamicFluxProcess, libecs::ContinuousProcess,
                         QuasiDynamicFluxProcessInterface )
{
public:

    LIBECS_DM_OBJECT( QuasiDynamicFluxProcess, libecs::Process );

    QuasiDynamicFluxProcess()
        : irreversible_( 0 ),
          vmax_( 0 )
    {
        theFluxDistributionVector.reserve( 0 );
    }

    ~QuasiDynamicFluxProcess()
    {
        ; // do nothing
    }

    SET_METHOD( libecs::Integer, Irreversible );

    virtual GET_METHOD( libecs::Integer, Irreversible );

    SET_METHOD( libecs::Real, Vmax );

    virtual GET_METHOD( libecs::Real, Vmax );

    SET_METHOD( libecs::Polymorph, FluxDistributionList );

    GET_METHOD( libecs::Polymorph, FluxDistributionList );

    virtual void initialize();

    virtual void fire();

    virtual libecs::VariableReferenceVector getFluxDistributionVector();

protected:

    libecs::VariableReferenceVector theFluxDistributionVector;
    libecs::Integer irreversible_;
    libecs::Real vmax_;
};

#endif /* __QUASIDYNAMICFLUXPROCESS_HPP */
