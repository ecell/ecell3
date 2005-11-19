//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2003 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// authors:
// Tatsuya Ishida
//
// E-Cell Project.
//


#include "ExpressionProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ExpressionAlgebraicProcess, ExpressionProcessBase )
{
 public:
  
  LIBECS_DM_OBJECT( ExpressionAlgebraicProcess, Process )
    {
      INHERIT_PROPERTIES( ExpressionProcessBase );
    }

  ExpressionAlgebraicProcess()
  {
    //FIXME: additional properties:
    // Unidirectional   -> call declareUnidirectional() in initialize()
    //                     if this is set
  }

  virtual ~ExpressionAlgebraicProcess()
  {
    ; // do nothing
  }
  
  
  virtual void fire()
    { 
      setActivity( theVirtualMachine.execute( theCompiledCode ) );
    }

};

LIBECS_DM_INIT( ExpressionAlgebraicProcess, Process );
