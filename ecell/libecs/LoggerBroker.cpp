//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#include "libecs.hpp"

#include "LoggerBroker.hpp"
#include "Logger.hpp"
#include "MessageInterface.hpp"
#include "PrimitiveType.hpp"
#include "FQPI.hpp"
#include "RootSystem.hpp"
#include "System.hpp"
#include "Substance.hpp"
#include "Reactor.hpp"

namespace libecs
{

  LoggerPtr LoggerBroker::getLogger( StringCref id_name, StringCref property_name )
  {
    const PairOfStrings p(id_name, property_name);
    LoggerMap::iterator position( theLoggerMap.find( p ) );
    if( position != theLoggerMap.end() )
      {
	return position->second;
      }
    else
      {
	appendLogger( id_name, property_name );
	position = theLoggerMap.find( p );
	return position->second;
      }
  }

  void LoggerBroker::appendLogger( StringCref fqpnstring, StringCref property_name )
  {
    FQPI aFQPI( fqpnstring );
    String aSystemPathString( aFQPI.getSystemPathString() );
    SystemPtr aSystemPtr = theRootSystem->getSystem( aSystemPathString );

    //    MessageInterfacePtr aMessageInterfacePtr = NULLPTR;
    PropertyMapIterator pmitr( NULLPTR );

    switch( aFQPI.getPrimitiveType() )
      {
      case SUBSTANCE:
	//	aMessageInterfacePtr = aSystemPtr->getSubstance( aFQPI.getIdString() );
	pmitr = aSystemPtr->getSubstance( aFQPI.getIdString() )->getMessageSlot( property_name );
    	break;
      case REACTOR:
	//	aMessageInterfacePtr = aSystemPtr->getReactor( aFQPI.getIdString() );
	pmitr = aSystemPtr->getReactor( aFQPI.getIdString() )->getMessageSlot( property_name );
	break;
      case SYSTEM:
	break;
      }
    //    PropertyMapIterator pmitr( aMessageInterfacePtr->getMessageSlot( property_name ) );

    const PairOfStrings p( fqpnstring, property_name );
    //    theLoggerMap[ p ] = new Logger( *pmitr->second->getProxy() );
    theLoggerMap.insert(PairInLoggerMap( p, new Logger( *pmitr->second->getProxy() ) ) );
  }
  

} // namespace libecs








