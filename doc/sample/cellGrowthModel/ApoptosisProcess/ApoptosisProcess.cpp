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
#include <iostream>
#include "ApoptosisProcess.hpp"
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>

using namespace std;
USE_LIBECS;
namespace python = boost::python;

LIBECS_DM_INIT( ApoptosisProcess, Process);

void ApoptosisProcess::initialize()
{

  PythonProcessBase::initialize();
  

  python::handle<> a( PyEval_EvalCode( (PyCodeObject*)
  				       theCompiledInitializeMethod.ptr(),
  				       theGlobalNamespace.ptr(), 
  				       theLocalNamespace.ptr() ) );
}


void ApoptosisProcess::fire()
{
  
  python::object a( python::eval( theExpression.c_str(),
                                  theGlobalNamespace, 
                                  theLocalNamespace ) );

  bool apoptosis = python::extract<bool>(a)();
  
  if (apoptosis) 
    {
      this->destroyCell();
    }
  
}
 

void ApoptosisProcess::destroyCell() 
{
  getModel()->removeEntity( getSuperSystem()->getFullID() );
}
