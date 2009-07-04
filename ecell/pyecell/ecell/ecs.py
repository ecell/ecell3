#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

import ecell._ecs

__all__ = (
    'getLibECSVersionInfo',
    'getLibECSVersion',
    'VariableReference',
    'Entity',
    'System',
    'Process',
    'Variable',
    'PropertyAttributes',
    'LoggerPolicy',
    'VariableReference',
    'VariableReferences',
    'Simulator'
    )

def __init__():
    import os
    if os.name != "nt":
        import sys
        try:
            import DLFCN
            
            # RTLD_GLOBAL is needed so that rtti across dynamic modules can work
            # RTLD_LAZY   may be needed so that the system can resolve dependency among
            #             dynamic modules after dlopened it
            
            sys.setdlopenflags( DLFCN.RTLD_LAZY | DLFCN.RTLD_GLOBAL )
        except:
            None

    for i in __all__:
        globals()[ i ] = getattr( ecell._ecs, i )

__init__()

class Simulator( ecell._ecs.Simulator ):
    def __init__( self, *args, **kwargs ):
        import ecell.config
        ecell._ecs.Simulator.__init__( self, *args, **kwargs )
        self.setDMSearchPath( self.DM_SEARCH_PATH_SEPARATOR.join( ecell.config.dm_path ) )

