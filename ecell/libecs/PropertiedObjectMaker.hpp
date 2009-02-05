//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef ___PROPERTIEDOBJECTMAKER_H___
#define ___PROPERTIEDOBJECTMAKER_H___

#include "PropertiedClass.hpp"
#include "dmtool/ModuleMaker.hpp"


namespace libecs
{
  
  /* *defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  class LIBECS_API PropertiedObjectMaker 
    : 
    public SharedModuleMaker<PropertiedClass>
  {
  public:
    PropertiedObjectMaker();
    virtual ~PropertiedObjectMaker();
  };

#define NewPropertiedObjectModule(CLASS) NewDynamicModule(PropertiedObject,CLASS)

  /** @} */ //end of libecs_module 

} // namespace libecs

#endif /* ___PROPERTIEDOBJECTMAKER_H___ */
