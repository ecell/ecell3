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

#ifndef __SHAREDMODULEMAKERINTERFACE_HPP
#define __SHAREDMODULEMAKERINTERFACE_HPP

#include <string>
#include <ltdl.h>

class SharedModuleMakerInterface
{
public:
    virtual ~SharedModuleMakerInterface() {}

    virtual void setSearchPath( const std::string& path ) = 0;

    virtual std::string getSearchPath() const = 0;

public:
    static const char PATH_SEPARATOR = LT_PATHSEP_CHAR;
};

#endif /* __SHAREDMODULEMAKERINTERFACE_HPP */
