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

#ifndef __DYNAMIC_MODULE_INFO_HPP
#define __DYNAMIC_MODULE_INFO_HPP

#include <string>
#include <utility>

#include "dmtool/DMObject.hpp"

/**
  DynamicModuleInfo defines an interface that provides the meta-information
  that can be used to annotate the module.
 */
class DM_IF DynamicModuleInfo
{
public:
    class EntryIterator
    {
    public:
        virtual ~EntryIterator() {}

        virtual bool next() = 0;

        virtual std::pair< std::string, const void* > current() = 0;
    };

public:
    virtual ~DynamicModuleInfo() {}
    virtual const void* getInfoField( std::string const& aFieldName ) const = 0;
    virtual EntryIterator* getInfoFields() const = 0;
};

#endif /* __DYNAMIC_MODULE_INFO_HPP */
