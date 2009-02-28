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

#ifndef __DMOBJECT_HPP
#define __DMOBJECT_HPP

#ifdef WIN32
#ifndef DM_IMPORTS
#define DM_IF __declspec(dllexport)
#else
#define DM_IF __declspec(dllimport)
#endif /* !DM_IMPORTS */
#else
#define DM_IF
#endif /* WIN32 */

#include "dmtool/DynamicModuleDescriptor.hpp"

#define DM_DESCRIPTOR_ENTRY( CLASSNAME ) \
    { \
        #CLASSNAME, \
        &CLASSNAME::createInstance, \
        &CLASSNAME::getClassInfoPtr, \
        &CLASSNAME::initializeModule, \
        &CLASSNAME::finalizeModule \
    }

#define DM_INIT( CLASSNAME )\
  extern "C"\
  {\
    DM_IF DynamicModuleDescriptor __dm_descriptor = DM_DESCRIPTOR_ENTRY( CLASSNAME ); \
  } // 

#define DM_NEW_STATIC( MAKER, BASE, CLASSNAME )\
  { \
    static DynamicModuleDescriptor desc = DM_DESCRIPTOR_ENTRY( CLASSNAME ); \
    ( MAKER )->addClass( new StaticDynamicModule< BASE >( desc ) ); \
  } //

#define DM_OBJECT( CLASSNAME )\
 static void* createInstance() { return new CLASSNAME ; }\


#define DM_BASECLASS( CLASSNAME )\
public:\
 typedef CLASSNAME * (* AllocatorFuncPtr )()


#endif /* __DMOBJECT_HPP */

