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
A program for supporting Eml object, ecell.eml.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'emlsupport'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import ecell.eml
import ecell.ecs
import ecell.emc
import ecell.Session

import os

import numpy

import PathwayProxy


class EmlSupport( ecell.eml.Eml ):


    def __init__( self, fileName=None, fileObject=None ):
        '''
        read EML file and set the file name
        fileName: (str) EML file name
        '''

        if fileObject == None and fileName == None:
            raise TypeError, ' fileObject(file) or fileName(str) must be set'
        
        if type( fileName ) == str:
            self.setEmlFileName( fileName )
            fileObject = open( self.getEmlFileName() )
            ecell.eml.Eml.__init__( self, fileObject )

        elif type( fileObject ) == str or type( fileObject ) == file:
            self.setEmlFileName( '' )
            ecell.eml.Eml.__init__( self, fileObject )
        else:
            raise TypeError, ' fileObject(file) or fileName(str) must be set'

    # end of __init__


    def createSession( self ):
        '''
        create and return the instance from file name
        return aSession
        '''
        
        aSimulator = ecell.emc.Simulator()
        aSession = ecell.Session.Session( aSimulator )
        # aSession.loadModel( self )

        ##
        ## Session can't recieve an ecell.eml.Eml instance, now
        ##

        ## WARNING!! : calling Session's private functions        
        aSession._Session__loadStepper( self )
        aSession._Session__loadEntity( self )
        aSession._Session__loadAllProperty( self )
        
        aSession.theModelName = self.getEmlFileName()
        
        return aSession

    # end of createSession


    def createPathwayProxy( self, indexList=None ):
        '''
        generate a PathwayProxy
        indexList: (list) a list of indices in processList
        return PathwayProxy
        '''

        processList = self.getProcessList()
        
        if not indexList:
            return PathwayProxy.PathwayProxy( self, processList )

        else:
            pathwayList = []
            for i in indexList:
                pathwayList.append( processList[ i ] )
            return PathwayProxy.PathwayProxy( self, pathwayList )

    # end of createPathwayProxy

    
    def setEmlFileName( self, fileName ):
        '''
        set the eml file name
        fileName: (str) an EML file name
        '''

        if type( fileName ) == str:
            self.__theEmlFileName = os.path.abspath( fileName )
        else:
            raise TypeError, ' The type of fileName must be string (file name) '

    # end of setEmlFileName


    def getEmlFileName( self ):
        '''
        simply return the eml file name handled now
        return theEmlFileName
        '''

        return self.__theEmlFileName

    # end of getEmlFileName
    

    def save( self, fileName=None ):
        '''
        save domtree as an EML file
        fileName: (str) an output file name
        when you don\'t set a file name, save it to self.getEmlFileName()
        '''

        if fileName:
            ecell.eml.Eml.save( self, fileName )
        else:
            ecell.eml.Eml.save( self, self.getEmlFileName() )

    # end of save
    

    def getAllEntityList( self, entityType, rootSystemPath ):
        '''
        get the list of all entities under the root system path
        entityType: (str) \'Variable\' or \'Process\' or \'System\'
        rootSystemPath: (str) the root system path
        return entityList
        '''
        
        entityList = self.getEntityList( entityType, rootSystemPath )

        size = len( entityList )
        for c in range( size ):
            entityList[ c ] = '%s:%s:%s' % ( entityType, rootSystemPath, entityList[ c ] )

        subSystemList = self.getEntityList( 'System', rootSystemPath )

        for subSystem in subSystemList:
            subSystemPath = ecell.ecssupport.joinSystemPath( rootSystemPath, subSystem )
            # recursive call
            entityList.extend( self.getAllEntityList( entityType, subSystemPath ) )

        entityList.sort()

        return entityList

    # end of getFullEntityList


    def getSystemList( self ):
        '''
        getAllEntityList( \'System\', \'/\' )
        '''
        
        systemList = self.getAllEntityList( 'System', '/' )
        systemList.append( 'System::/' )

        return systemList

    # end of getSystemList
                                        

    def getVariableList( self ):
        '''
        getAllEntityList( \'Variable\', \'/\' )
        '''
        
        return self.getAllEntityList( 'Variable', '/' )

    # end of getVariableList


    def getProcessList( self ):
        '''
        getAllEntityList( \'Process\', \'/\' )
        '''
        
        return self.getAllEntityList( 'Process', '/' )

    # end of getProcessList


    def writeEntityProperty( self, fullIDString, value ):
        '''
        '''

        fullID = fullIDString.split( ':' )
        if ( len( fullID ) == 4 ):
            self.deleteEntityProperty( fullID[ 0 ] + ':' + fullID[ 1 ] + ':' + fullID[ 2 ], fullID[ 3 ] )
            self.setEntityProperty( fullID[ 0 ] + ':'+ fullID[ 1 ] + ':' + fullID[ 2 ], fullID[ 3 ], [ '%.8e' % value ] )

    # end of writeEntityProperty
    

# end of EmlSupport


if __name__ == '__main__':

    from emlsupport import EmlSupport

    import sys
    import os


    def main( fileName ):
        
        anEmlSupport = EmlSupport( fileName )

        print 'system fullID list ='
        print anEmlSupport.getAllEntityList( 'System', '/' )
        print 'variable fullID list ='
        print anEmlSupport.getVariableList()
        print 'process fullID list ='
        print anEmlSupport.getProcessList()

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
