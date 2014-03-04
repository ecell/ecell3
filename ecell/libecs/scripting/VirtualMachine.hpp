//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
//
// authors:
//   Koichi Takahashi
//   Tatsuya Ishida
//   Yasuhiro Naito
//
// E-Cell Project.
//

#ifndef __VIRTUALMACHNE_HPP
#define __VIRTUALMACHNE_HPP

#include "libecs/libecs.hpp"
#include "libecs/scripting/ExpressionCompiler.hpp"
#include "libecs/AssocVector.h"
#include <map>
#include "libecs/Model.hpp"

namespace libecs { namespace scripting
{

class LIBECS_API VirtualMachine
{
public:
  
    VirtualMachine()
    : theModel( 0 )
    {
      // ; do nothing
    }
  
    ~VirtualMachine() {}
  
    const libecs::Real execute( Code const& aCode );
    
    const libecs::Real getDelayedValue( libecs::Integer x, libecs::Real t );
    
    void setModel( Model* const aModel )
    {
        theModel = aModel;
    }

private:
    typedef std::map< Real, Real > TimeSeries;
    typedef Loki::AssocVector< Integer, TimeSeries > DelayMap;
    DelayMap theDelayMap;
    
    libecs::Model*  theModel;

};

} } // namespace libecs::scripting

#endif /* __VIRTUALMACHNE_HPP */
