//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
// Contact information:
//   Nathan Addy, Research Associate     Voice: 510-981-8748
//   The Molecular Sciences Institute    Email: addy@molsci.org  
//   2168 Shattuck Ave.                  
//   Berkeley, CA 94704
//
//END_HEADER

#include <libecs/Model.hpp>
#include <sstream>
#include <string>
#include "ApoptosisProcess.hpp"

LIBECS_DM_INIT( ApoptosisProcess, Process);

ApoptosisProcess::ApoptosisProcess()
  :
  expressionChecksConcentration( true ),
  apoptosisThreshold( 0.0f )
{
}


SET_METHOD_DEF( String, Type, ApoptosisProcess)
{
  if (value == String("Concentration") )
    {
      expressionChecksConcentration = true;
    }
  else if (value == String("Population"))
    {
      expressionChecksConcentration = false;
    }
  else throw 0;
}

GET_METHOD_DEF( String, Type, ApoptosisProcess )
{
  if (expressionChecksConcentration)
    {
      return String("Concentration");
    }
  else
    {
      return String("Population");
    }
}


SET_METHOD_DEF( String, GreaterOrLessThan, ApoptosisProcess )
{
  if (value[0] == '<' )
    {
      apoptosisBelowThreshold = true;
    }
  else if (value[0] == '>')
    {
      apoptosisBelowThreshold = false;
    }
  else throw 0;
}

GET_METHOD_DEF( String, GreaterOrLessThan, ApoptosisProcess )
{
  if( apoptosisBelowThreshold )
    {
      return String("<");
    }
  else
    {
      return String(">");
    }
}


SET_METHOD_DEF( Real, Expression, ApoptosisProcess )
{
  apoptosisThreshold = value;
}

GET_METHOD_DEF( Real, Expression, ApoptosisProcess )
{
  return apoptosisThreshold;
}


void ApoptosisProcess::initialize()
{
}


void ApoptosisProcess::fire()
{
  // Check to see if the value of the variable is greater or less than the 
  // threshold.  

  destroyCell();
}


void ApoptosisProcess::destroyCell()
{
  SystemPtr theParentSystem( cellToBeKilled->getParent() );

  this->getModel()->removeSystem( cellToBeKilled );
}
