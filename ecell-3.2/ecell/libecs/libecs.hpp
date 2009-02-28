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

#ifndef __LIBECS_HPP
#define __LIBECS_HPP

#include "libecs/Defs.hpp"
#include "dmtool/ModuleMaker.hpp"
#include "EcsObject.hpp"

namespace libecs
{

class EcsObject;

LIBECS_API extern int const MAJOR_VERSION;
LIBECS_API extern int const MINOR_VERSION;
LIBECS_API extern int const MICRO_VERSION;

LIBECS_API extern char const* const VERSION_STRING;


LIBECS_API const int getMajorVersion();
LIBECS_API const int getMinorVersion();
LIBECS_API const int getMicroVersion();
LIBECS_API const std::string getVersion();

LIBECS_API bool initialize();
LIBECS_API void finalize();
LIBECS_API ModuleMaker< EcsObject >* createDefaultModuleMaker();

} // namespace libecs

#endif // __LIBECS_HPP
