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

#ifndef __DISCRETEEVENTPROCESS_HPP
#define __DISCRETEEVENTPROCESS_HPP


#include "libecs.hpp"
#include "Process.hpp"

namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


  DECLARE_VECTOR( DiscreteEventProcessPtr, DiscreteEventProcessVector );

  /**

  */

  LIBECS_DM_CLASS( DiscreteEventProcess, Process )
  {

  public:

    LIBECS_DM_OBJECT_ABSTRACT( DiscreteEventProcess )
      {
	INHERIT_PROPERTIES( Process );

	PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, StepInterval );

	PROPERTYSLOT( Polymorph, DependentProcessList,
		      NULLPTR,
		      &DiscreteEventProcess::getDependentProcessList );
      }


  public:

    DiscreteEventProcess()
      :
      theStepInterval( 0.0 ),
      Index( -1 )
      {
	; // do nothing
      }

    virtual ~DiscreteEventProcess();

    SIMPLE_SET_GET_METHOD( Int, Index );

    GET_METHOD( Real, StepInterval )
      {
	return theStepInterval;
      }

    virtual GET_METHOD( Real, TimeScale )
      {
	return getStepInterval();
      }

    virtual void initialize()
      {
	Process::initialize();
      }
    
    // virtual void process();

    virtual void updateStepInterval() = 0;

    void clearDependentProcessVector()
      {
	theDependentProcessVector.clear();
      }
    
    void addDependentProcess( DiscreteEventProcessPtr aProcessPtr );

    /**
       Check if this Process can affect on the given Process.

       This dependency is 

    */

    virtual const bool 
      checkProcessDependency( DiscreteEventProcessPtr 
			      anDiscreteEventProcessPtr ) const;

    DiscreteEventProcessVectorCref getDependentProcessVector() const
      {
	return theDependentProcessVector;
      }

    const Polymorph getDependentProcessList() const;


  protected:

    DiscreteEventProcessVector theDependentProcessVector;

    Real theStepInterval;

    Int Index;

  };





  /*@}*/

} // namespace libecs

#endif /* __DISCRETEEVENTPROCESS_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
