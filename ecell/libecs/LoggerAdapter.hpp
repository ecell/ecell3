//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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


#if !defined(__LOGGERADAPTER_HPP)
#define __LOGGERADAPTER_HPP

#include "libecs.hpp"

namespace libecs
{


  /** @addtogroup logging The Data Logging Module.
      The Data Logging Module.

      @ingroup libecs
      
      @{ 
   */ 

  /** @file */

  class LoggerAdapter
  {

  public:

    virtual ~LoggerAdapter();

    virtual const Real getValue() const = 0;

  protected:

    LoggerAdapter();

  };


  /** @} */ // logging module

} // namespace libecs


#endif /* __LOGGERADAPTER_HPP */

