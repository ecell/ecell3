//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

#ifndef __KOYURUGI_H
#define __KOYURUGI_H
#include "Defs.hpp"

namespace libecs
{


  // classes

  DECLARE_CLASS( System );
  DECLARE_CLASS( AccumulatorMaker );
  DECLARE_CLASS( Accumulator );
  DECLARE_CLASS( SimpleAccumulator );
  DECLARE_CLASS( RoundDownAccumulator );
  DECLARE_CLASS( RoundOffAccumulator );
  DECLARE_CLASS( ReserveAccumulator );
  DECLARE_CLASS( MonteCarloAccumulator );
  DECLARE_CLASS( Environment );
  DECLARE_CLASS( Monolithic );
  DECLARE_CLASS( Cytoplasm );
  DECLARE_CLASS( Membrane );
  DECLARE_CLASS( Cell );
  DECLARE_CLASS( Entity );
  DECLARE_CLASS( SystemPath );
  DECLARE_CLASS( FQID );
  DECLARE_CLASS( FQPI );
  DECLARE_CLASS( Integrator );
  DECLARE_CLASS( Euler1Integrator );
  DECLARE_CLASS( RungeKutta4Integrator );
  DECLARE_CLASS( Reactant );
  DECLARE_CLASS( Reactor );
  DECLARE_CLASS( isRegularReactor );
  DECLARE_CLASS( ReactorMaker );
  DECLARE_CLASS( RootSystem );
  DECLARE_CLASS( Stepper );
  DECLARE_CLASS( MasterStepper );
  DECLARE_CLASS( StepperLeader );
  DECLARE_CLASS( SlaveStepper );
  DECLARE_CLASS( Euler1Stepper );
  DECLARE_CLASS( RungeKutta4Stepper );
  DECLARE_CLASS( StepperMaker );
  DECLARE_CLASS( Substance );
  DECLARE_CLASS( SubstanceMaker );
  DECLARE_CLASS( System );
  DECLARE_CLASS( isRegularReactorItem );
  DECLARE_CLASS( SystemMaker );
  DECLARE_CLASS( Message );
  DECLARE_CLASS( AbstractMessageSlotClass );
  DECLARE_CLASS( MessageSlotClass );
  DECLARE_CLASS( MessageInterface );
  DECLARE_CLASS( UniversalVariable );
  DECLARE_CLASS( LoggerBroker );
  DECLARE_CLASS( Logger );
  DECLARE_CLASS( DataPoint );

  // exceptions

  DECLARE_CLASS( Exception );
  DECLARE_CLASS( UnexpectedError );
  DECLARE_CLASS( NotFound );
  DECLARE_CLASS( CantOpen );
  DECLARE_CLASS( BadID );
  DECLARE_CLASS( MessageException );
  DECLARE_CLASS( CallbackFailed );
  DECLARE_CLASS( BadMessage );
  DECLARE_CLASS( NoMethod );
  DECLARE_CLASS( NoSlot );
  DECLARE_CLASS( InvalidPrimitiveType );



} // namespace libecs


#endif // __KOYURUGI_H


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
