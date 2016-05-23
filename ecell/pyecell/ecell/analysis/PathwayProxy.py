#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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
A program for handling and defining a pathway.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'PathwayProxy'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__coyright__ = ''
__license__ = ''


import ecell.eml
from ecell.ecssupport import *
from ecell.analysis.util import createVariableReferenceFullID

import copy

import numpy


class PathwayProxy:

    def __init__( self, anEmlSupport, processList=None ):
        '''
        anEmlSupport: Eml support object
        processList: (list) a list of process full path
        '''

        self.theEmlSupport = anEmlSupport

        if processList:
            self.setProcessList( processList )
        else:
            self.setProcessList( [] )

    # end of __init__


    def setProcessList( self, processList ):
        '''
        set and detect a pathway
        processList: (list) a list of process full ID
        '''

        # check the existence of processes,
        # and create relatedVariableList
        self.__processList = []
        self.__variableList = []

        for processFullID in processList:
            
#            if not self.theEmlSupport.isEntityExist( processFullID ):
#                continue

            self.__processList.append( processFullID )
            
            try:
                aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

            except AttributeError, e:
                continue
            
            for aVariableReference in aVariableReferenceList:
                fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
                fullIDString = ecell.ecssupport.createFullIDString( fullID )

                if self.__variableList.count( fullIDString ) == 0:
                    self.__variableList.append( fullIDString )

        self.__processList.sort()
        self.__variableList.sort()

    # end of setProcessList


    def getProcessList( self ):
        '''
        return processList
        '''

        return copy.copy( self.__processList )

    # end of getProcessList
    

    def addProcess( self, processFullID ):
        '''
        add a process to the pathway
        processFullID: (str) a process full ID
        '''

        if not self.__processList.count( processFullID ) == 0:
            return
#        elif not ecell.eml.Eml.isEntityExist( processFullID ):
#            return

        # add process
        self.__processList.append( processFullID )
        self.__processList.sort()

        # update the related variable list
        try:
            aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

        except AttributeError, e:
            return
        
        for aVariableReference in aVariableReferenceList:
            fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
            fullIDString = ecell.ecssupport.createFullIDString( fullID )

            if self.__variableList.count( fullIDString ) == 0:
                self.__variableList.append( fullIDString )

        self.__variableList.sort()

    # end of addProcess

    
    def removeProcess( self, processIndexList ):
        '''
        remove processes from the pathway
        processIndexList: (list) a list of indices of processes
        '''

        indexList = copy.copy( processIndexList )
        indexList.sort()
        indexList.reverse()

        removedProcessList = []
        for i in indexList:
            if len( self.__processList ) > i:
                removedProcessList.append( self.__processList.pop( i ) )

        removedVariableList = []
        for processFullID in removedProcessList:

#            if not ecell.eml.Eml.isEntityExist( self.theEmlSupport, processFullID ):
#                continue
            
            try:
                aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

            except AttributeError, e:
                continue
            
            for aVariableReference in aVariableReferenceList:
                fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
                fullIDString = ecell.ecssupport.createFullIDString( fullID )

                if removedVariableList.count( fullIDString ) == 0:
                    removedVariableList.append( fullIDString )

        for processFullID in self.__processList:

#            if not self.theEmlSupport.isEntityExist( processFullID ):
#                continue
            
            try:
                aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

            except AttributeError, e:
                continue
            
            for aVariableReference in aVariableReferenceList:
                fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
                fullIDString = ecell.ecssupport.createFullIDString( fullID )

                if not removedVariableList.count( fullIDString ) == 0:
                    removedVariableList.remove( fullIDString )

        for variableFullID in removedVariableList:
            self.__variableList.remove( variableFullID )

    # end of removeProcess


    def take( self, processIndexList ):
        '''
        create and return a sub-pathway
        processIndexList: (list) a list of indices of processes
        return PathwayProxy
        '''

        processList = []
        for i in processIndexList:
            if len( self.__processList ) > i:
                processList.append( self.__processList[ i ] )

        subPathway = PathwayProxy( self.theEmlSupport, processList )
        return subPathway

    # end of removeProcess


    def getVariableList( self ):
        '''
        return relatedVariableList
        '''

        return copy.copy( self.__variableList )

    # end of getVariableList


    def removeVariable( self, variableIndexList ):
        '''
        remove variables from the pathway
        variableIndexList: (list) a list of indices of variables
        '''

        indexList = copy.copy( variableIndexList )
        indexList.sort()
        indexList.reverse()

        for i in indexList:
            if len( self.__variableList ) > i:
                self.__variableList.pop( i )

    # end of removeVariable


    def addVariable( self, variableFullID ):
        '''
        recover a removed variable to the pathway
        variableFullID: (str) a variable full ID
        '''

        if not self.__variableList.count( variableFullID ) == 0:
            return 1
#        elif not ecell.eml.Eml.isEntityExist( variableFullID ):
#            return 0

        for processFullID in self.__processList:

            try:
                aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

            except AttributeError, e:
                continue
            
            for aVariableReference in aVariableReferenceList:
                fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
                fullIDString = fullID[ 1 ] + ':' + fullID[ 2 ]

                if fullIDString == variableFullID:
                    self.__variableList.append( variableFullID )
                    self.__variableList.sort()
                    return 1

        return 0
        
    # end of addProcess

    
    def getIncidentMatrix( self, mode=0 ):
        '''
        create the incident matrix (array)
        mode: (0 or 1) 0 means that only the \'write\' variables are checked. 0 is set as default.
        return incidentMatrix
        '''

        incidentMatrix = numpy.zeros( ( len( self.__variableList ), len( self.__processList ) ) )

        for j in range( len( self.__processList ) ):

            processFullID = self.__processList[ j ]

            try:
                aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

            except AttributeError, e:
                continue
                
            for aVariableReference in aVariableReferenceList:
                fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
                fullIDString = ecell.ecssupport.createFullIDString( fullID )

                try:
                    i = self.__variableList.index( fullIDString )
                except ValueError:
                    # should some warning message be showed?
                    continue

                if mode:
                    if len( aVariableReference ) > 2:
                        coeff = int( aVariableReference[ 2 ] )
                        if coeff != 0:
                            incidentMatrix[ i ][ j ] = 1
                else:
                    incidentMatrix[ i ][ j ] = 1

        return incidentMatrix
    
    # end of getIncidentMatrix


    def getStoichiometryMatrix( self ):
        '''
        create the stoichiometry matrix (array)
        return stoichiometryMatrix
        '''

        stoichiometryMatrix = numpy.zeros( ( len( self.__variableList ), len( self.__processList ) ), float )

        for j in range( len( self.__processList ) ):

            processFullID = self.__processList[ j ]

            try:
                aVariableReferenceList = self.theEmlSupport.getEntityProperty( processFullID + ':VariableReferenceList' )

            except AttributeError, e:
                continue
            
            for aVariableReference in aVariableReferenceList:
                fullID = createVariableReferenceFullID( aVariableReference[ 1 ], processFullID )
                fullIDString = ecell.ecssupport.createFullIDString( fullID )

                try:
                    i = self.__variableList.index( fullIDString )
                except ValueError:
                    # should some warning message be showed?
                    continue

                if len( aVariableReference ) > 2:
                    coeff = int( aVariableReference[ 2 ] )
                    if coeff != 0:
                        stoichiometryMatrix[ i ][ j ] += coeff

        return stoichiometryMatrix
    
    # end of getStoichiometryMatrix


    def getReversibilityList( self ):
        '''
        check and return the reversibilities (isReversible) for processes
        default value is 0, irreversible
        return reversibilityList
        '''

        reversibilityList = []
        for processFullID in self.__processList:

            propertyList = self.theEmlSupport.getEntityPropertyList( processFullID )

            if propertyList.count( 'isReversible' ) != 0:
                # isReversible is handled as float
                isReversible = float( self.theEmlSupport.getEntityProperty( processFullID + ':isReversible' )[ 0 ] )
                reversibilityList.append( int( isReversible ) )
                
            else:
                # default value, irreversible
                reversibilityList.append( 0 )

        return reversibilityList
    
    # end of getReversibilityList


# end of PathwayProxy


if __name__ == '__main__':

    from emlsupport import EmlSupport

    import sys
    import os


    def main( filename ):

        anEmlSupport = EmlSupport( filename )
        pathwayProxy = anEmlSupport.createPathwayProxy()

        print 'process list ='
        print pathwayProxy.getProcessList()
        print 'related variable list ='
        print pathwayProxy.getVariableList()
        print 'incident matrix ='
        print pathwayProxy.getIncidentMatrix()
        print 'stoichiometry matrix ='
        print pathwayProxy.getStoichiometryMatrix()
        print 'reversibility list ='
        print pathwayProxy.getReversibilityList()

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
