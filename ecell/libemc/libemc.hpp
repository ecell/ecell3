//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//


#ifndef __LIBEMC_HPP
#define __LIBEMC_HPP

#include <functional>

#include "libecs/libecs.hpp"

namespace libemc
{

  /** @addtogroup libemc_module The E-Cell Micro Core Interface (libemc)
   * EMC module.
   * @{ 
   */ 
  
  /** @file */

  DECLARE_CLASS( EventChecker );
  DECLARE_CLASS( EventHandler );

  DECLARE_CLASS( Simulator );
  DECLARE_CLASS( SimulatorImplementation );



  DECLARE_RCPTR( EventChecker );
  DECLARE_RCPTR( EventHandler );

  class EventHandler
    :
    public std::unary_function<void,void> 
  {
  public:
    EventHandler() {}
    virtual ~EventHandler() {}

    virtual void operator()( void ) const = 0;
  };

  class EventChecker
    :
    public std::unary_function<bool,void>
  {
  public:
    EventChecker() {}
    virtual ~EventChecker() {}

    virtual bool operator()( void ) const = 0;
  };

  class DefaultEventChecker
    :
    public EventChecker
  {
  public:
    DefaultEventChecker() {}
    //    virtual ~DefaultEventChecker() {}

    virtual bool operator()( void ) const
    {
      return false;
    }
  };




  /** @} */ //end of libemc_module 

} // namespace libemc

#endif   /* __LIBEMC_HPP */
