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

#ifndef __SRMREACTOR_HPP
#define __SRMREACTOR_HPP

#include "Reactor.hpp"
#include "PropertySlotMaker.hpp"

namespace libecs
{

  DECLARE_CLASS( SRMReactor );

  class SRMReactor
    :
    public Reactor
  {

  public:

    class PriorityCompare
    {
    public:
      bool operator()( SRMReactorPtr aLhs, SRMReactorPtr aRhs ) const
      {
	return ( aLhs->getPriority() < aRhs->getPriority() );
      }
    };

    SRMReactor() 
      :
      thePriority( 0 )
    {
      makeSlots();  
    } 
    
    virtual ~SRMReactor()
    {
      ; // do nothing
    }

    virtual void initialize()
    {
      Reactor::initialize();
    }

    virtual void react()
    {
      ; // do nothing
    }

    void setPriority( RealCref aValue )
    {
      thePriority = static_cast<Int>( aValue );
    }

    const Real getPriority() const
    {
      return static_cast<Real>( thePriority );
    }


  protected:

    void makeSlots();

  private:

    Int         thePriority;

  };



  inline void SRMReactor::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Priority", *this, 
				      Type2Type<Real>(), // Int?
				      &SRMReactor::setPriority,
				      &SRMReactor::getPriority ) );
  }



} // namespace libecs


#endif /* __SRMREACTOR_HPP */
