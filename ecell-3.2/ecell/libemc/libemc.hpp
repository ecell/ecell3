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

#ifndef __LIBEMC_HPP
#define __LIBEMC_HPP

#include <functional>

#ifdef DLL_EXPORT
#undef DLL_EXPORT
#define _DLL_EXPORT
#endif /* DLL_EXPORT */

#include "libecs/libecs.hpp"

#ifdef _DLL_EXPORT
#define DLL_EXPORT
#undef _DLL_EXPORT
#endif /* _DLL_EXPORT */

// WIN32 stuff
#if defined( WIN32 )

#if defined( LIBEMC_EXPORTS ) || defined( DLL_EXPORT )
#define LIBEMC_API __declspec(dllexport)
#else
#define LIBEMC_API __declspec(dllimport)
#endif /* LIBEMC_EXPORTS */

#else
#define LIBEMC_API
#endif /* WIN32 */

namespace libemc
{

DECLARE_CLASS( EventChecker );
DECLARE_CLASS( EventHandler );

DECLARE_CLASS( Simulator );
DECLARE_CLASS( SimulatorImplementation );

DECLARE_SHAREDPTR( EventChecker );
DECLARE_SHAREDPTR( EventHandler );

class LIBEMC_API EventHandler
    : public std::unary_function< void, void >
{
public:
    EventHandler() {}
    virtual ~EventHandler() {}

    virtual void operator()( void ) const = 0;
};

class LIBEMC_API EventChecker
    : public std::unary_function< bool, void >
{
public:
    EventChecker() {}
    virtual ~EventChecker() {}

    virtual bool operator()( void ) const = 0;
};

class LIBEMC_API DefaultEventChecker
    :
    public EventChecker
{
public:
    DefaultEventChecker() {}
    //        virtual ~DefaultEventChecker() {}

    virtual bool operator()( void ) const
    {
        return false;
    }
};

} // namespace libemc

#endif /* __LIBEMC_HPP */
