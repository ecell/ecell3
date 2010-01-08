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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "VariableMaker.hpp"


namespace libecs
{
  VariableMaker::VariableMaker( PropertiedObjectMaker& maker )
    : theBackend( maker )
  {
    makeClassList();
  }

  VariableMaker::~VariableMaker()
  {
    ; // do nothing
  }

  Variable* VariableMaker::make( const std::string& aClassName )
  {
    const PropertiedObjectMaker::SharedModule& mod(
	theBackend.getModule( aClassName, false ) );
    if ( mod.getTypeName() != "Variable" )
      {
	throw TypeError( "specified class is not a Variable" );
      }
    return reinterpret_cast< Variable* >( theBackend.make( aClassName ) );
  }

  const PropertiedObjectMaker::SharedModule& VariableMaker::getModule(
      const std::string& aClassName, bool forceReload )
  {
    const PropertiedObjectMaker::SharedModule& mod(
	theBackend.getModule( aClassName, forceReload ) );
    if ( mod.getTypeName() != "Variable" )
      {
	throw TypeError( "specified class is not a Variable" );
      }
    return mod;
  }

  void VariableMaker::makeClassList()
  {
    theBackend.NewDynamicModule( PropertiedClass, Variable );
  }

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
