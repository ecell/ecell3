
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
        filename = '../../../../doc/sample/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
