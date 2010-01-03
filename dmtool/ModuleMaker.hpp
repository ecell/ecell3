//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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


#ifndef __MODULEMAKER_HPP
#define __MODULEMAKER_HPP

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <cassert>

#include "DynamicModule.hpp"

/**
    ModuleMaker is used to instantiate
    various subclasses of certain template parameter class T. 
*/

template<class T >
class ModuleMaker 
{

public:

    typedef DynamicModule< T > Module;
    typedef std::map< const std::string, Module* > ModuleMap;
    typedef typename ModuleMap::iterator ModuleMapIterator;

public: 

    ModuleMaker()
    {
        ; // do nothing
    }

    virtual ~ModuleMaker()
    {
        for( ModuleMapIterator i = this->theModuleMap.begin();
             i != this->theModuleMap.end(); ++i )
        {
            delete i->second;
        }
    }


    virtual const Module& getModule( const std::string& aClassName, bool forceReload = false )
    {
        if ( this->theModuleMap.find( aClassName ) == this->theModuleMap.end() )
        {
            throw DMException( "Can't find module [" + aClassName + "]." );
        }
        return *this->theModuleMap[ aClassName ];
    }


    /**
         Add a class to the subclass list.
         @param dm a pointer to a DynamicModule to be added
    */

    virtual void addClass( Module* dm )
    {
        assert( dm != NULL && dm->getModuleName() != NULL );
        this->theModuleMap[ dm->getModuleName() ] = dm;
    }

    virtual const ModuleMap& getModuleMap() const
    {
        return theModuleMap;
    }

protected:

    ModuleMap theModuleMap;

};

#endif /* __MODULEMAKER_HPP */
