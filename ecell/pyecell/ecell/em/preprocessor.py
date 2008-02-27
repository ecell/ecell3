#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

"""
A program for converting .em file to EML.
This program is part of E-Cell Simulation Environment Version 3.
"""

__program__ = 'emparser'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Koichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2003 Keio University'
__license__ = 'GPL'

import em

__all__ = (
    'preprocess'
    )

class Hook(em.Hook):
    def __init__( self ):
        self.needs_linecontrol = False

    def lineControl( self, file, line ):
        self.interpreter.write( '%%line %d %s\n' % ( line, file ) )

    def afterInclude( self ):
        self.lineControl( *self.interpreter.context().identify() )

    def beforeIncludeHook( self, name, file, locals ):  
        self.lineControl( name, 1 )  

    def afterExpand( self, result ):
        self.needs_linecontrol = True 

    def afterEvaluate( self, result ):
        self.needs_linecontrol = True

    def afterSignificate( self ):
        self.needs_linecontrol = 1
                         
    def atParse( self, scanner, locals ):
        if not self.needs_linecontrol:
            return
        self.lineControl( *self.interpreter.context().identify() )
        self.needs_linecontrol = 0        

def preprocess( input, output ):
    interp = em.Interpreter( output = output,
                             hooks = ( Hook(), ) )
    interp.flatten()
    if hasattr( input, 'name' ):
        filename = input.name
    else:
        filename = '<string>'
    interp.wrap( interp.file, ( input, filename ) )
    interp.flush()
