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

#ifndef __LIBECS_HPP
#define __LIBECS_HPP
#include "Defs.hpp"
#include "RCPtr.hpp"


namespace libecs
{

  /** @defgroup libecs The Libecs library
   * The libecs library
   * @{ 
   */ 
  
  /** @file */

  // Declarations to be exported


  DECLARE_LIST  ( String, StringList );
  DECLARE_VECTOR( String, StringVector );

  DECLARE_RCPTR( StringList );
  DECLARE_RCPTR( StringVector );


  // classes

  DECLARE_CLASS( System );
  DECLARE_CLASS( AccumulatorMaker );
  DECLARE_CLASS( Accumulator );
  DECLARE_CLASS( Entity );
  DECLARE_CLASS( EntityType );
  DECLARE_CLASS( SystemPath );
  DECLARE_CLASS( FullID );
  DECLARE_CLASS( FullPN );
  DECLARE_CLASS( Integrator );
  DECLARE_CLASS( Reactant );
  DECLARE_CLASS( Reactor );
  DECLARE_CLASS( ReactorMaker );
  DECLARE_CLASS( Stepper );
  DECLARE_CLASS( MasterStepper );
  DECLARE_CLASS( Model );
  DECLARE_CLASS( SlaveStepper );
  DECLARE_CLASS( StepperMaker );
  DECLARE_CLASS( Substance );
  DECLARE_CLASS( SubstanceMaker );
  DECLARE_CLASS( System );
  DECLARE_CLASS( SystemMaker );
  DECLARE_CLASS( PropertySlot );
  DECLARE_CLASS( ProxyPropertySlot );
  DECLARE_CLASS( PropertyInterface );
  DECLARE_CLASS( PropertySlotMaker );
  DECLARE_CLASS( Polymorph );
  DECLARE_CLASS( LoggerBroker );
  DECLARE_CLASS( Logger );
  DECLARE_CLASS( DataPoint );
  DECLARE_CLASS( DataPointVector );
  DECLARE_CLASS( PhysicalLogger );

  // containers

  DECLARE_VECTOR( Polymorph, PolymorphVector );


  // exceptions

  DECLARE_CLASS( Exception );
  DECLARE_CLASS( UnexpectedError );
  DECLARE_CLASS( NotFound );
  DECLARE_CLASS( CantOpen );
  DECLARE_CLASS( BadID );
  DECLARE_CLASS( CallbackFailed );
  DECLARE_CLASS( NoMethod );
  DECLARE_CLASS( NoSlot );
  DECLARE_CLASS( InvalidEntityType );



  // reference counted pointer types

  DECLARE_RCPTR( PolymorphVector );
  DECLARE_RCPTR( DataPointVector );
  
  /** @} */ 

} // namespace libecs

#endif // __LIBECS_HPP


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
