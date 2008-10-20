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
// modified by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "PropertyInterface.hpp"
#include "Exceptions.hpp"

#include "EcsObject.hpp"

namespace libecs
{


  ///////////////////////////// EcsObject
  const PropertyAttributes EcsObject::
  defaultGetPropertyAttributes( StringCref aPropertyName ) const
  {
    THROW_EXCEPTION( NoSlot, 
		     getClassName() + 
		     String( ": No property slot [" )
		     + aPropertyName + "].  Get property attributes failed." );
  }

  const StringVector&
  EcsObject::defaultGetPropertyList() const
  {
    static StringVector emptyVector;
    return emptyVector;
  }
  
  void EcsObject::defaultSetProperty( StringCref aPropertyName, 
					    PolymorphCref aValue )
  {
    THROW_EXCEPTION( NoSlot,
		     getClassName() + 
		     String( ": No property slot [" )
		     + aPropertyName + "].  Set property failed." );
  }

  const Polymorph 
  EcsObject::defaultGetProperty( StringCref aPropertyName ) const
  {
    THROW_EXCEPTION( NoSlot, 
		     getClassName() + 
		     String( ": No property slot [" )
		     + aPropertyName + "].  Get property failed." );
  }
  
  void EcsObject::registerLogger( LoggerPtr aLoggerPtr )
  {
    if( std::find( theLoggerVector.begin(), theLoggerVector.end(), aLoggerPtr )
	== theLoggerVector.end() )
      {
   	theLoggerVector.push_back( aLoggerPtr );
      }
  }

  void EcsObject::removeLogger( LoggerPtr aLoggerPtr )
  {
    LoggerVectorIterator i( find( theLoggerVector.begin(), 
				  theLoggerVector.end(),
				  aLoggerPtr ) );
    
    if( i != theLoggerVector.end() )
      {
	theLoggerVector.erase( i );
      }

  }


  void EcsObject::throwNotSetable()
  {
    THROW_EXCEPTION( AttributeError, "Not setable." );
  }

  void EcsObject::throwNotGetable()
  {
    THROW_EXCEPTION( AttributeError, "Not getable." );
  }

  StringCref EcsObject::getClassName() const
  {
     return getPropertyInterface().getClassName();
  }

#define NULLSET_SPECIALIZATION_DEF( TYPE )\
  template <> void EcsObject::nullSet<TYPE>( Param<TYPE>::type )\
  {\
    throwNotSetable();\
  } //

  NULLSET_SPECIALIZATION_DEF( Real );
  NULLSET_SPECIALIZATION_DEF( Integer );
  NULLSET_SPECIALIZATION_DEF( String );
  NULLSET_SPECIALIZATION_DEF( Polymorph );

#define NULLGET_SPECIALIZATION_DEF( TYPE )\
  template <> const TYPE EcsObject::nullGet<TYPE>() const\
  {\
    throwNotGetable();\
    return TYPE(); \
  } //

  NULLGET_SPECIALIZATION_DEF( Real );
  NULLGET_SPECIALIZATION_DEF( Integer );
  NULLGET_SPECIALIZATION_DEF( String );
  NULLGET_SPECIALIZATION_DEF( Polymorph );


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/