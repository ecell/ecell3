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
#include "FullID.hpp"
#include "RootSystem.hpp"
#include "System.hpp"
#include "Substance.hpp"
#include "Reactor.hpp"


namespace libecs
{

  LoggerPtr LoggerBroker::getLogger( FullPropertyNameCref fpn )
  {
    LoggerMapIterator aLoggerMapIterator( theLoggerMap.find( fpn ) );
    if( aLoggerMapIterator != theLoggerMap.end() )
      {
	return aLoggerMapIterator->second;
      }
    else
      {
	appendLogger( fpn );
	aLoggerMapIterator = theLoggerMap.find( fpn );
	return aLoggerMapIterator->second;
      }
  }

  void LoggerBroker::appendLogger( FullPropertyNameCref fpn )
  {
    SystemPtr aSystemPtr = theRootSystem->getSystem( fpn.getSystemPath() );

    PropertyMapIterator aPropertyMapIterator( NULLPTR );

    String anID( fpn.getID() );
    String aPropertyName( fpn.getPropertyName() );

    switch( fpn.getPrimitiveType() )
      {
      case SUBSTANCE:
	aPropertyMapIterator = aSystemPtr->getSubstance( anID )->
	  getMessageSlot( aPropertyName );
    	break;
      case REACTOR:
	aPropertyMapIterator = aSystemPtr->getReactor( anID )->
	  getMessageSlot( aPropertyName );
	break;
      case SYSTEM:
	aPropertyMapIterator = aSystemPtr->getSystem( anID )->
	  getMessageSlot( aPropertyName );
	break;
      default:
	throw BadID( __PRETTY_FUNCTION__,
		     "Bad primitive type" );
      }

    LoggerPtr aLoggerPtr = new Logger();
    aPropertyMapIterator->second->getProxy()->setLogger( aLoggerPtr );
    theLoggerMap[fpn] = aLoggerPtr;
  }
  

} // namespace libecs








