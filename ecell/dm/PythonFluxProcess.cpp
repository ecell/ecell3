//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2003 Keio University
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
// authors:
// Kouichi Takahashi <shafi@e-cell.org>
// Nayuta Iwata
//
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "FullID.hpp"
#include "PythonFluxProcess.hpp"

USE_LIBECS;


LIBECS_DM_INIT( PythonFluxProcess, Process );

  
void PythonFluxProcess::fire()
{
  python::handle<> 
    aHandle( PyEval_EvalCode( reinterpret_cast<PyCodeObject*>
			      ( theCompiledExpression.ptr() ),
			      theGlobalNamespace.ptr(), 
			      theLocalNamespace.ptr() ) );

  python::object aResultObject( aHandle );
  
  // do not use extract<double> for efficiency
  if( ! PyFloat_Check( aResultObject.ptr() ) )
    {
      THROW_EXCEPTION( SimulationError, 
		       "[" + getFullID().getString() + 
		       "]: The expression gave a non-float object." );
    }

  const Real aFlux( PyFloat_AS_DOUBLE( aResultObject.ptr() ) );

  setFlux( aFlux );
}
