//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "VariableReference.hpp"
#include "FullID.hpp"

#include "DiscreteEventProcess.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( DiscreteEventProcess, Process );


  DiscreteEventProcess::~DiscreteEventProcess()
  {
    ; // do nothing
  }

  const Polymorph DiscreteEventProcess::getDependentProcessList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theDependentProcessVector.size() );
    
    for ( DiscreteEventProcessVectorConstIterator 
	    i( theDependentProcessVector.begin() );
	  i != theDependentProcessVector.end(); ++i ) 
      {
	DiscreteEventProcessPtr anDiscreteEventProcess( *i );
	
	FullIDCref aFullID( anDiscreteEventProcess->getFullID() );
	const String aFullIDString( aFullID.getString() );
	
	aVector.push_back( aFullIDString );
      }
    
    return aVector;
  }


  const bool DiscreteEventProcess::
  checkProcessDependency( DiscreteEventProcessPtr 
			  anDiscreteEventProcessPtr ) const
  {
    VariableReferenceVectorCref 
      aVariableReferenceVector( anDiscreteEventProcessPtr->
				getVariableReferenceVector() );
    
    for( VariableReferenceVectorConstIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end() ; ++i )
      {
	VariableReferenceCref aVariableReference1(*i );
	
	VariableCptr const aVariable1( aVariableReference1.getVariable() );

	for( VariableReferenceVectorConstIterator 
	       j( aVariableReferenceVector.begin() );
	     j != aVariableReferenceVector.end(); ++j )
	  {
	    VariableReferenceCref aVariableReference2( *j );
	    VariableCptr const aVariable2( aVariableReference2.getVariable() );
	    const Int aCoefficient2( aVariableReference2.getCoefficient() );
	    
	    if( aVariable1 == aVariable2 && aCoefficient2 < 0 )
	      {
		return true;
	      }
	  }
      }

    return false;
  }

  void DiscreteEventProcess::
  addDependentProcess( DiscreteEventProcessPtr aProcessPtr )
  {
    if( std::find( theDependentProcessVector.begin(), 
		   theDependentProcessVector.end(), aProcessPtr ) 
	== theDependentProcessVector.end() )
      {
	theDependentProcessVector.push_back( aProcessPtr );

	// optimization: sort by memory address
	std::sort( theDependentProcessVector.begin(), 
		   theDependentProcessVector.end() );
      }

  }




} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
