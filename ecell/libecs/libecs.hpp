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

#ifndef __LIBECS_HPP
#define __LIBECS_HPP

#include "libecs/Defs.hpp"

#include <list>
#include <vector>
#include <map>

namespace libecs
{

  /** @defgroup libecs The Libecs library
   * The libecs library
   * @{ 
   */ 
  
  /** @file */


  LIBECS_API extern int const MAJOR_VERSION;
  LIBECS_API extern int const MINOR_VERSION;
  LIBECS_API extern int const MICRO_VERSION;

  LIBECS_API extern char const* const VERSION_STRING;


  LIBECS_API const int getMajorVersion();
  LIBECS_API const int getMinorVersion();
  LIBECS_API const int getMicroVersion();
  LIBECS_API const std::string getVersion();

  LIBECS_API bool initialize();
  LIBECS_API void finalize();
  // LIBECS_API const String getLoadedDMList(); // XXX: not implemented

  // Forward declarations.


  // string STL containers.
  typedef std::list<String> StringList;
  typedef std::vector<String> StringVector;
  typedef std::map<const String, String, std::less<const String> > StringMap;
  DECLARE_SHAREDPTR( StringList );
  DECLARE_SHAREDPTR( StringVector );


  // classes

  DECLARE_CLASS( System );
  DECLARE_CLASS( Entity );
  DECLARE_CLASS( EntityType );
  DECLARE_CLASS( SystemPath );
  DECLARE_CLASS( FullID );
  DECLARE_CLASS( FullPN );
  DECLARE_CLASS( VariableReference );
  DECLARE_CLASS( Process );
  DECLARE_CLASS( DiscreteEventProcess );
  DECLARE_CLASS( ProcessMaker );
  DECLARE_CLASS( Stepper );
  DECLARE_CLASS( SystemStepper );
  DECLARE_CLASS( Interpolant );
  DECLARE_CLASS( Model );
  DECLARE_CLASS( Scheduler );
  DECLARE_CLASS( StepperEvent );
  DECLARE_CLASS( StepperMaker );
  DECLARE_CLASS( Variable );
  DECLARE_CLASS( VariableMaker );
  DECLARE_CLASS( System );
  DECLARE_CLASS( SystemMaker );
  DECLARE_CLASS( PropertySlotBase );
  DECLARE_CLASS( PropertyInterfaceBase );
  DECLARE_CLASS( PropertiedClass );
  DECLARE_CLASS( PropertySlotProxy );
  DECLARE_CLASS( Polymorph );
  DECLARE_CLASS( LoggerBroker );
  DECLARE_CLASS( Logger );
  DECLARE_CLASS( LoggerAdapter );
  DECLARE_CLASS( DataPoint );
  DECLARE_CLASS( LongDataPoint );
  DECLARE_CLASS( DataPointAggregator );
  DECLARE_CLASS( DataPointVector );


  // containers
  DECLARE_VECTOR( Polymorph,    PolymorphVector );
  DECLARE_VECTOR( VariablePtr,  VariableVector );
  DECLARE_VECTOR( ProcessPtr,   ProcessVector );
  DECLARE_VECTOR( SystemPtr,    SystemVector );
  DECLARE_VECTOR( StepperPtr,   StepperVector );
  DECLARE_VECTOR( LoggerPtr,    LoggerVector );
  //  DECLARE_VECTOR( PropertySlotPtr, PropertySlotVector );

  // exceptions

  DECLARE_CLASS( Exception );
  DECLARE_CLASS( UnexpectedError );
  DECLARE_CLASS( NotFound );
  DECLARE_CLASS( BadID );
  DECLARE_CLASS( CallbackFailed );
  DECLARE_CLASS( NoMethod );
  DECLARE_CLASS( NoSlot );
  DECLARE_CLASS( InvalidEntityType );

  DECLARE_MAP ( const String, Polymorph, std::less<const String>,
				PolymorphMap);


  // other reference counted pointer types

  DECLARE_SHAREDPTR( PolymorphVector );
  DECLARE_SHAREDPTR( DataPointVector );
  
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
