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

#ifndef __ECELL3_DM_HPP
#define __ECELL3_DM_HPP


// these macros assume that ECELL3_DM_TYPE and ECELL3_DM_CLASSNAME macros
// are already defined before use.
//
// ECELL3_DM_TYPE and ECELL3_DM_CLASSNAME macros must be defined 
// *after* this file is included.
//
// if _ECELL3_DM_TYPE or _ECELL3_DM_CLASSNAME is defined *before*
// including this file, it is used as a default value of
// ECELL3_DM_TYPE or ECELL3_DM_CLASSNAME macro.


#define ECELL3_DM_OBJECT\
 LIBECS_DM_OBJECT( ECELL3_DM_TYPE, ECELL3_DM_CLASSNAME )

#define ECELL3_DM_INIT\
 DM_INIT( ECELL3_DM_TYPE, ECELL3_DM_CLASSNAME )

#define ECELL3_DM_CLASS\
 class ECELL3_DM_CLASSNAME


#ifdef _ECELL3_DM_CLASSNAME
#    define ECELL3_DM_CLASSNAME _ECELL3_DM_CLASSNAME
#    undef  _ECELL3_DM_CLASSNAME
#endif

#ifdef _ECELL3_DM_TYPE
#    define ECELL3_DM_TYPE _ECELL3_DM_TYPE
#    undef  _ECELL3_DM_TYPE
#endif


#define ECELL3_CREATE_PROPERTYSLOT_SET_GET( TYPE, NAME )\
CREATE_PROPERTYSLOT_SET_GET( TYPE, NAME, ECELL3_DM_CLASSNAME )

#define ECELL3_CREATE_PROPERTYSLOT_SET    ( TYPE, NAME )\
CREATE_PROPERTYSLOT_SET    ( TYPE, NAME, ECELL3_DM_CLASSNAME )

#define ECELL3_CREATE_PROPERTYSLOT_GET    ( TYPE, NAME )\
CREATE_PROPERTYSLOT_GET    ( TYPE, NAME, ECELL3_DM_CLASSNAME )



#endif /* __ECELL3_DM_HPP */
