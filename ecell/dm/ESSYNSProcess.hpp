//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
//END_HEADER
#ifndef __ESSYNSPROCESS_HPP
#define __ESSYNSPROCESS_HPP

#include <vector>

#include "libecs.hpp"
#include "Process.hpp"

USE_LIBECS;


LIBECS_DM_CLASS( ESSYNSProcess, Process )
{
  
 public:

  DECLARE_VECTOR( libecs::Real, RealVector );


  LIBECS_DM_OBJECT_ABSTRACT( ESSYNSProcess )
    {
      INHERIT_PROPERTIES( Process );  
    }
  
  ESSYNSProcess()
    {
      ;
    }

  virtual ~ESSYNSProcess()
    {
      ;
    }

  virtual void initialize()
    {
      Process::initialize();
    }
    
  virtual void fire()
    {
      ;
    }
    
  virtual const std::vector<RealVector>& getESSYNSMatrix() = 0;

  virtual GET_METHOD( Integer, SystemSize ) = 0;
    
 protected:

};

LIBECS_DM_INIT_STATIC( ESSYNSProcess, Process );


#endif /* __ESSYNSPROCESS_HPP */
