//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org> for
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//


#include <signal.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "libecs/Exceptions.hpp"

#include "libecs/Process.hpp"
#include "libecs/VariableReference.hpp"

#include "PyEcs.hpp"

using namespace libecs;


// exception translators

//void translateException( libecs::ExceptionCref anException )
//{
//  PyErr_SetString( PyExc_RuntimeError, anException.what() );
//}

void translateException( const std::exception& anException )
{
  PyErr_SetString( PyExc_RuntimeError, anException.what() );
}


static PyObject* getLibECSVersionInfo()
{
  PyObject* aPyTuple( PyTuple_New( 3 ) );
    
  PyTuple_SetItem( aPyTuple, 0, PyInt_FromLong( libecs::getMajorVersion() ) );
  PyTuple_SetItem( aPyTuple, 1, PyInt_FromLong( libecs::getMinorVersion() ) );
  PyTuple_SetItem( aPyTuple, 2, PyInt_FromLong( libecs::getMicroVersion() ) );
  
  return aPyTuple;
}

BOOST_PYTHON_MODULE( _ecs )
{
  using namespace boost::python;

  // without this it crashes when Logger::getData() is called. why?
  import_array();

  // functions

  def( "getLibECSVersionInfo", &getLibECSVersionInfo );
  def( "getLibECSVersion",     &libecs::getVersion );

  def( "setDMSearchPath", &libecs::setDMSearchPath );
  def( "getDMSearchPath", &libecs::getDMSearchPath );
  //  def( "getDMInfoList",   &libemc::getDMInfoList );
  //  def( "getDMInfo",       &libemc::getDMInfo );


  to_python_converter< Polymorph, Polymorph_to_python >();
  to_python_converter< DataPointVectorSharedPtr, 
    DataPointVectorSharedPtr_to_python >();
  to_python_converter< PolymorphMap, PolymorphMap_to_python>();
  
  register_Polymorph_from_python();

  register_exception_translator<Exception>     ( &translateException );
  register_exception_translator<std::exception>( &translateException );



  class_<VariableReference>( "VariableReference", no_init )

    // properties
    .add_property( "Coefficient",
		   &VariableReference::getCoefficient )   // read-only
    .add_property( "MolarConc", 
		   &VariableReference::getMolarConc ) // read-only
    .add_property( "Name", 
		   &VariableReference::getName ) // read-only
    .add_property( "NumberConc", 
		   &VariableReference::getNumberConc ) // read-only
    .add_property( "IsFixed",
    		   &VariableReference::isFixed )      // read-only
    .add_property( "IsAccessor",
    		   &VariableReference::isAccessor )       // read-only
    .add_property( "TotalVelocity",
		   &VariableReference::getTotalVelocity )
    .add_property( "Value", 
		   &VariableReference::getValue, 
		   &VariableReference::setValue )
    .add_property( "Velocity",
		   &VariableReference::getVelocity )

    // methods
    .def( "addFlux",     &VariableReference::addFlux )
    .def( "addValue",    &VariableReference::addValue )
    .def( "addVelocity", &VariableReference::addVelocity )
    .def( "getSuperSystem",  // this should be a property, but not supported
	  &VariableReference::getSuperSystem,
	  python::return_value_policy<python::reference_existing_object>() )
    ;

  class_<Process, bases<>, Process, boost::noncopyable>
    ( "Process", no_init )

    // properties
    .add_property( "Activity",
		   &Process::getActivity,
		   &Process::setActivity )
    .add_property( "Priority",
		   &Process::getPriority )
    .add_property( "StepperID",
		   &Process::getStepperID )

    // methods
    .def( "addValue",    &Process::addValue )
    .def( "getPositiveVariableReferenceOffset",     
	  &Process::getPositiveVariableReferenceOffset )
    .def( "getSuperSystem",   // this can be a property, but not supported
	  &Process::getSuperSystem,
	  python::return_value_policy<python::reference_existing_object>() )
    .def( "getVariableReference",       // this should be a property
	  &Process::getVariableReference,
	  python::return_internal_reference<>() )
    .def( "getVariableReferenceVector",       // this should be a property
	  &Process::getVariableReferenceVector,
	  python::return_value_policy<python::reference_existing_object>() )
    .def( "getZeroVariableReferenceOffset",     
	  &Process::getZeroVariableReferenceOffset )
    .def( "setFlux",     &Process::setFlux )
    ;


  class_<System, bases<>, System, boost::noncopyable>( "System", no_init )

    // properties
    .add_property( "Size",
		   &System::getSize )
    .add_property( "SizeN_A",
		   &System::getSizeN_A )
    .add_property( "StepperID",
		   &System::getStepperID )
    // methods
    .def( "getSuperSystem",   // this should be a property, but not supported
	  &System::getSuperSystem,
	  python::return_value_policy<python::reference_existing_object>() )
    ;



  class_<VariableReferenceVector, bases<>, VariableReferenceVector>
    ( "VariableReferenceVector" )

    .def( vector_indexing_suite<VariableReferenceVector>() )
    ;


}

