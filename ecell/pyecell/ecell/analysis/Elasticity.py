#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
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
A program for calculating elasticities
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'Elasticity'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


from util import RELATIVE_PERTURBATION, ABSOLUTE_PERTURBATION, allzero, createIndependentGroupList

import numpy


def getElasticityArray( pathwayProxy, fullPN ):
    '''
    calculate and return the elasticities (array)
    with 1st order Taylor expansion
    pathwayProxy: a PathwayProxy instance
    fullPN: (str) the full property name
    return elasticityArray
    '''

    processList = pathwayProxy.getProcessList()

    size = len( processList )
    
    # first step
    elasticityArray = numpy.zeros( size, float )
    
    aSession = pathwayProxy.theEmlSupport.createSession()
    aSession.theSimulator.initialize()
    for i in range( size ):
        elasticityArray[ i ] = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )
    
    # second step
    aSession = pathwayProxy.theEmlSupport.createSession()

    value = aSession.theSimulator.getEntityProperty( fullPN )
    aPerturbation = RELATIVE_PERTURBATION * value + ABSOLUTE_PERTURBATION

    aSession.theSimulator.setEntityProperty( fullPN, value + aPerturbation )
    aSession.theSimulator.initialize()

    for c in range( size ):
        elasticityArray[ c ] = ( aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' ) - elasticityArray[ c ] ) / aPerturbation

    return elasticityArray

# end of getElasticityArray


def getAcculateElasticityArray( pathwayProxy, fullPN ):
    '''
    calculate and return the elasticities (array)
    with 2nd order Taylor expansion
    pathwayProxy: a PathwayProxy instance
    fullPN: (str) the full property name
    return elasticityArray
    '''

    processList = pathwayProxy.getProcessList()
    size = len( processList )

    elasticityArray = numpy.zeros( size, float )
    
    # first step
    aSession = pathwayProxy.theEmlSupport.createSession()

    value = aSession.theSimulator.getEntityProperty( fullPN )
    aPerturbation = RELATIVE_PERTURBATION * value + ABSOLUTE_PERTURBATION

    aSession.theSimulator.setEntityProperty( fullPN, value - 2.0 * aPerturbation )
    aSession.theSimulator.initialize()

    for c in range( size ):
        elasticityArray[ c ] = aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

    # second step    
    aSession = pathwayProxy.theEmlSupport.createSession()
    aSession.theSimulator.setEntityProperty( fullPN, value - aPerturbation )
    aSession.theSimulator.initialize()
    for c in range( size ):
        elasticityArray[ c ] -= 8.0 * aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

    # third step    
    aSession = pathwayProxy.theEmlSupport.createSession()
    aSession.theSimulator.setEntityProperty( fullPN, value + aPerturbation )
    aSession.theSimulator.initialize()
    for c in range( size ):
        elasticityArray[ c ] += 8.0 * aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

    # last(fourth) step
    aSession = pathwayProxy.theEmlSupport.createSession()
    aSession.theSimulator.setEntityProperty( fullPN, value + 2.0 * aPerturbation )
    aSession.theSimulator.initialize()
    for c in range( size ):
        elasticityArray[ c ] -= aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

        elasticityArray[ c ] /= 12.0 * aPerturbation

    return elasticityArray

# end of getAcculateElasticityArray


def getElasticity( pathwayProxy, fullPN ):
    '''
    default getElasticity function
    now this is equal to getElasticityArray( pathwayProxy, fullPN ):
    '''

    return getElasticityArray( pathwayProxy, fullPN )

# end of getElasticity


def getEpsilonElasticity( pathwayProxy, variableFullID ):
    '''
    default getEpsilonElasticity function
    variableFullID: the full ID of the variable
    now this is equal to getElasticityArray( pathwayProxy, fullPN ):
    '''

    return getElasticity( pathwayProxy, variableFullID + ':Value' )

# end of getElasticity


def convertToScaled( pathwayProxy, fullPN, elasticityArray ):
    '''
    convert an elasticity (dict) or (array) to the scaled elasticity
    pathwayProxy: a PathwayProxy instance
    fullPN: (str) the full property name
    elasticityArray: the elasticity (array)
    return the scaled elasticity (array)
    '''

    processList = pathwayProxy.getProcessList()
    size = len( elasticityArray )
    scaledElasticityArray = numpy.zeros( size, float )

    aSession = pathwayProxy.theEmlSupport.createSession()
    try:
        value = float( pathwayProxy.theEmlSupport.getEntityProperty( fullPN )[ -1 ] )
    except:
        value = aSession.theSimulator.getEntityProperty( fullPN )

    aSession.theSimulator.initialize()

    for i in range( size ):

        anActivity = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )
        
        if anActivity != 0.0:
            scaledElasticityArray[ i ] = elasticityArray[ i ] * value / anActivity
        else:
            # zero division
            scaledElasticityArray[ i ] = 0.0

    return scaledElasticityArray

# end of convertToScaled


def getScaledElasticity( pathwayProxy, fullPN ):
    '''
    calculate and return the scaled elasticities as (array)
    refer getElasticityArray( pathwayProxy, fullPN )
    '''
    
    elasticityArray = getElasticity( pathwayProxy, fullPN )
    return convertToScaled( pathwayProxy, fullPN, elasticityArray )

# end of getScaledElasticityArray


def getScaledEpsilonElasticity( pathwayProxy, variableFullID ):
    '''
    default getEpsilonElasticity function
    variableFullID: the full ID of the variable
    now this is equal to getScaledElasticity( pathwayProxy, fullPN ):
    '''

    return getScaledElasticity( pathwayProxy, variableFullID + ':Value' )

# end of getElasticity


def getEpsilonElasticityMatrix( pathwayProxy ):
    '''
    calculate and return the epsilon elasticities (matrix)
    pathwayProxy: a PathwayProxy instance
    return elasticityMatrix
    '''

    variableList = pathwayProxy.getVariableList()

    variableFullPNList = []
    for variableFullID in variableList:
        variableFullPNList.append( variableFullID + ':Value' )

    return getElasticityMatrix( pathwayProxy, variableFullPNList )

# end of getEpsilonElasticityMatrix


def getElasticityMatrix( pathwayProxy, fullPNList ):
    '''
    calculate and return the elasticities (matrix)
    pathwayProxy: a PathwayProxy instance
    fullPNList: (list) a list of property names
    return elasticityMatrix
    '''
    
    processList = pathwayProxy.getProcessList()

    elasticityMatrix = numpy.zeros( ( len( fullPNList ), len( processList ) ), float )

    for i in range( len( fullPNList ) ):
        elasticityArray = getElasticity( pathwayProxy, fullPNList[ i ] )
        numpy.put( elasticityMatrix[ i ], range( len( processList ) ), elasticityArray )

    return elasticityMatrix

# end of getElasticityMatrix


def getScaledEpsilonElasticityMatrix( pathwayProxy ):
    '''
    calculate and return the scaled epsilon elasticities (matrix)
    pathwayProxy: a PathwayProxy instance
    return scaledElasticityMatrix
    '''

    variableList = pathwayProxy.getVariableList()

    variableFullPNList = []
    for variableFullID in variableList:
        variableFullPNList.append( variableFullID + ':Value' )

    return getScaledElasticityMatrix( pathwayProxy, variableFullPNList )

# end of getScaledEpsilonElasticityMatrix


def getScaledElasticityMatrix( pathwayProxy, fullPNList ):
    '''
    calculate and return the scaled elasticities (matrix)
    pathwayProxy: a PathwayProxy instance
    fullPNList: (list) of property name
    return scaledElasticityMatrix
    '''

    processList = pathwayProxy.getProcessList()

    scaledElasticityMatrix = numpy.zeros( ( len( fullPNList ), len( processList ) ), float )

    for i in range( len( fullPNList ) ):
        scaledElasticityArray = getScaledElasticity( pathwayProxy, fullPNList[ i ] )
        numpy.put( scaledElasticityMatrix[ i ], range( len( processList ) ), scaledElasticityArray )

    return scaledElasticityMatrix

# end of getScaledEpsilonElasticityMatrix


def getEpsilonElasticityMatrix2( pathwayProxy ):
    '''
    calculate and return the elasticities (matrix)
    pathwayProxy: a PathwayProxy instance
    return elasticityMatrix
    '''

    variableList = pathwayProxy.getVariableList()
    processList = pathwayProxy.getProcessList()

    elasticityMatrix = numpy.zeros( ( len( variableList ), len( processList ) ), float )

    incidentMatrix = pathwayProxy.getIncidentMatrix()
    independentGroupList = createIndependentGroupList( incidentMatrix )
    
    activityBuffer = numpy.zeros( len( processList ), float )

    aSession = pathwayProxy.theEmlSupport.createSession()

    aSession.theSimulator.initialize()
    for i in range( len( processList ) ):
        activityBuffer[ i ] = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )
    
    for groupList in independentGroupList:

        aSession = pathwayProxy.theEmlSupport.createSession()

        perturbationList = []
        for i in groupList:
            fullPN = variableList[ i ] + ':Value'
            value = aSession.theSimulator.getEntityProperty( fullPN )
            aPerturbation = RELATIVE_PERTURBATION * value + ABSOLUTE_PERTURBATION
            perturbationList.append( aPerturbation )
            aSession.theSimulator.setEntityProperty( fullPN, value + aPerturbation )

        aSession.theSimulator.initialize()

        for c in range( len( groupList ) ):
            i = groupList[ c ]
            aPerturbation = perturbationList[ c ]
            for j in range( len( processList ) ):
                if incidentMatrix[ i ][ j ]:
                    elasticityMatrix[ i ][ j ] = ( aSession.theSimulator.getEntityProperty( processList[ j ] + ':Activity' ) - activityBuffer[ j ] ) / aPerturbation

    return elasticityMatrix

# end of getEpsilonElasticityMatrix2


if __name__ == '__main__':

    from emlsupport import EmlSupport
    from Elasticity import *

    import sys
    import os


    def main( fileName, fullPN=None ):
        
        anEmlSupport = EmlSupport( fileName )
        pathwayProxy = anEmlSupport.createPathwayProxy()

        if fullPN != None:

            print 'elasticity array for \'%s\' =' % ( fullPN )
            print getElasticityArray( pathwayProxy, fullPN )
            print 'acculate elasticity array for \'%s\' =' % ( fullPN )
            print getAcculateElasticityArray( pathwayProxy, fullPN )
            print 'scaled elasticity array for \'%s\' =' % ( fullPN )
            print getScaledElasticity( pathwayProxy, fullPN )

        print 'epsilon elasticity matrix ='
        print getEpsilonElasticityMatrix( pathwayProxy )
        print 'scaled epsilon elasticity matrix ='
        print getScaledEpsilonElasticityMatrix( pathwayProxy )
        print 'epsilon elasticity matrix ='
        print getEpsilonElasticityMatrix2( pathwayProxy )

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ), 'Variable:/CELL/CYTOPLASM:A13P2G:Value' )
