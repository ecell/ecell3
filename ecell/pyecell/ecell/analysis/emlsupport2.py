#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
An efficient Eml support program, extending emlsupport.EmlSupport.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'emlsupport2'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import copy

import emlsupport


class EmlSupport2( emlsupport.EmlSupport ):


    def __init__( self, fileName=None, fileObject=None ):
        '''
        read EML file and set the file name
        fileName: (str) EML file name
        '''

        emlsupport.EmlSupport.__init__( self, fileName, fileObject )
        self.thePathwayProxy = None

        self.thePathwayProxy = self.createPathwayProxy()

    # end of __init__


    def getVariableList( self ):
        '''
        get the list of all variables in the Eml object
        return self.thePathwayProxy.getVariableList()
        '''

        if not self.thePathwayProxy:
            return emlsupport.EmlSupport.VariableList( self )
        else:
            return copy.copy( self.thePathwayProxy.getVariableList() )

    # end of getVariableList


    def getProcessList( self ):
        '''
        get the list of all processes in the Eml object
        return self.thePathwayProxy.getProcessList()
        '''

        if not self.thePathwayProxy:
            return emlsupport.EmlSupport.getProcessList( self )
        else:
            return copy.copy( self.thePathwayProxy.getProcessList() )

    # end of getProcessList


# end of EmlSupport2


if __name__ == '__main__':

    from emlsupport2 import EmlSupport2

    import sys
    import os


    def main( fileName ):

        anEmlSupport2 = EmlSupport2( fileName )
        
        print 'variable fullID list ='
        print anEmlSupport2.getVariableList()
        print 'process fullID list ='
        print anEmlSupport2.getProcessList()

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
