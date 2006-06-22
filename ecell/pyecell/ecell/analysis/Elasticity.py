
"""
A program for calculating elasticities
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'Elasticity'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import string

from util import RELATIVE_PERTURBATION, ABSOLUTE_PERTURBATION, allzero, createIndependentGroupList

import numpy


def getElasticityArray( aPathwayProxy, fullPN ):
    '''
    calculate and return the elasticities (array)
    with 1st order Taylor expansion
    aPathwayProxy: a PathwayProxy instance
    fullPN: (str) the full property name
    return elasticityArray
    '''

    processList = aPathwayProxy.getProcessList()

    size = len( processList )
    
    # first step
    elasticityArray = numpy.zeros( size, numpy.Float )
    
    aSession = aPathwayProxy.theEmlSupport.createInstance()
    aSession.step()
    for i in range( size ):
        elasticityArray[ i ] = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )
    
    # second step
    aSession = aPathwayProxy.theEmlSupport.createInstance()

    value = aSession.theSimulator.getEntityProperty( fullPN )
    aPerturbation = RELATIVE_PERTURBATION * value + ABSOLUTE_PERTURBATION

    aSession.theSimulator.setEntityProperty( fullPN, value + aPerturbation )
    aSession.step()

    for c in range( size ):
        elasticityArray[ c ] = ( aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' ) - elasticityArray[ c ] ) / aPerturbation

    return elasticityArray

# end of getElasticityArray


def getAcculateElasticityArray( aPathwayProxy, fullPN ):
    '''
    calculate and return the elasticities (array)
    with 2nd order Taylor expansion
    aPathwayProxy: a PathwayProxy instance
    fullPN: (str) the full property name
    return elasticityArray
    '''

    processList = aPathwayProxy.getProcessList()
    size = len( processList )

    elasticityArray = numpy.zeros( size, numpy.Float )
    
    # first step
    aSession = aPathwayProxy.theEmlSupport.createInstance()

    value = aSession.theSimulator.getEntityProperty( fullPN )
    aPerturbation = RELATIVE_PERTURBATION * value + ABSOLUTE_PERTURBATION

    aSession.theSimulator.setEntityProperty( fullPN, value - 2.0 * aPerturbation )
    aSession.step()

    for c in range( size ):
        elasticityArray[ c ] = aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

    # second step    
    aSession = aPathwayProxy.theEmlSupport.createInstance()
    aSession.theSimulator.setEntityProperty( fullPN, value - aPerturbation )
    aSession.step()
    for c in range( size ):
        elasticityArray[ c ] -= 8.0 * aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

    # third step    
    aSession = aPathwayProxy.theEmlSupport.createInstance()
    aSession.theSimulator.setEntityProperty( fullPN, value + aPerturbation )
    aSession.step()
    for c in range( size ):
        elasticityArray[ c ] += 8.0 * aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

    # last(fourth) step
    aSession = aPathwayProxy.theEmlSupport.createInstance()
    aSession.theSimulator.setEntityProperty( fullPN, value + 2.0 * aPerturbation )
    aSession.step()
    for c in range( size ):
        elasticityArray[ c ] -= aSession.theSimulator.getEntityProperty( processList[ c ] + ':Activity' )

        elasticityArray[ c ] /= 12.0 * aPerturbation

    return elasticityArray

# end of getAcculateElasticityArray


def getElasticity( aPathwayProxy, fullPN ):
    '''
    default getElasticity function
    now this is equal to getElasticityArray( aPathwayProxy, fullPN ):
    '''

    return getElasticityArray( aPathwayProxy, fullPN )

# end of getElasticity


def getEpsilonElasticity( aPathwayProxy, variableFullID ):
    '''
    default getEpsilonElasticity function
    variableFullID: the full ID of the variable
    now this is equal to getElasticityArray( aPathwayProxy, fullPN ):
    '''

    return getElasticity( aPathwayProxy, variableFullID + ':Value' )

# end of getElasticity


def convertToScaled( aPathwayProxy, fullPN, elasticityArray ):
    '''
    convert an elasticity (dict) or (array) to the scaled elasticity
    aPathwayProxy: a PathwayProxy instance
    fullPN: (str) the full property name
    elasticityArray: the elasticity (array)
    return the scaled elasticity (array)
    '''

    processList = aPathwayProxy.getProcessList()
    size = len( elasticityArray )
    scaledElasticityArray = numpy.zeros( size, numpy.Float )

    aSession = aPathwayProxy.theEmlSupport.createInstance()
    try:
        value = string.atof( aPathwayProxy.theEmlSupport.getEntityProperty( fullPN )[ -1 ] )
    except:
        value = aSession.theSimulator.getEntityProperty( fullPN )

    aSession.step()

    for i in range( size ):

        anActivity = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )
        
        if anActivity != 0.0:
            scaledElasticityArray[ i ] = elasticityArray[ i ] * value / anActivity
        else:
            # zero division
            scaledElasticityArray[ i ] = 0.0

    return scaledElasticityArray

# end of convertToScaled


def getScaledElasticity( aPathwayProxy, fullPN ):
    '''
    calculate and return the scaled elasticities as (array)
    refer getElasticityArray( aPathwayProxy, fullPN )
    '''
    
    elasticityArray = getElasticity( aPathwayProxy, fullPN )
    return convertToScaled( aPathwayProxy, fullPN, elasticityArray )

# end of getScaledElasticityArray


def getScaledEpsilonElasticity( aPathwayProxy, variableFullID ):
    '''
    default getEpsilonElasticity function
    variableFullID: the full ID of the variable
    now this is equal to getScaledElasticity( aPathwayProxy, fullPN ):
    '''

    return getScaledElasticity( aPathwayProxy, variableFullID + ':Value' )

# end of getElasticity


def getEpsilonElasticityMatrix( aPathwayProxy ):
    '''
    calculate and return the epsilon elasticities (matrix)
    aPathwayProxy: a PathwayProxy instance
    return elasticityMatrix
    '''

    variableList = aPathwayProxy.getVariableList()

    variableFullPNList = []
    for variableFullID in variableList:
        variableFullPNList.append( variableFullID + ':Value' )

    return getElasticityMatrix( aPathwayProxy, variableFullPNList )

# end of getEpsilonElasticityMatrix


def getElasticityMatrix( aPathwayProxy, fullPNList ):
    '''
    calculate and return the elasticities (matrix)
    aPathwayProxy: a PathwayProxy instance
    fullPNList: (list) a list of property names
    return elasticityMatrix
    '''
    
    processList = aPathwayProxy.getProcessList()

    elasticityMatrix = numpy.zeros( ( len( fullPNList ), len( processList ) ), numpy.Float )

    for i in range( len( fullPNList ) ):
        elasticityArray = getElasticity( aPathwayProxy, fullPNList[ i ] )
        numpy.put( elasticityMatrix[ i ], range( len( processList ) ), elasticityArray )

    return elasticityMatrix

# end of getElasticityMatrix


def getScaledEpsilonElasticityMatrix( aPathwayProxy ):
    '''
    calculate and return the scaled epsilon elasticities (matrix)
    aPathwayProxy: a PathwayProxy instance
    return scaledElasticityMatrix
    '''

    variableList = aPathwayProxy.getVariableList()

    variableFullPNList = []
    for variableFullID in variableList:
        variableFullPNList.append( variableFullID + ':Value' )

    return getScaledElasticityMatrix( aPathwayProxy, variableFullPNList )

# end of getScaledEpsilonElasticityMatrix


def getScaledElasticityMatrix( aPathwayProxy, fullPNList ):
    '''
    calculate and return the scaled elasticities (matrix)
    aPathwayProxy: a PathwayProxy instance
    fullPNList: (list) of property name
    return scaledElasticityMatrix
    '''

    processList = aPathwayProxy.getProcessList()

    scaledElasticityMatrix = numpy.zeros( ( len( fullPNList ), len( processList ) ), numpy.Float )

    for i in range( len( fullPNList ) ):
        scaledElasticityArray = getScaledElasticity( aPathwayProxy, fullPNList[ i ] )
        numpy.put( scaledElasticityMatrix[ i ], range( len( processList ) ), scaledElasticityArray )

    return scaledElasticityMatrix

# end of getScaledEpsilonElasticityMatrix


def getEpsilonElasticityMatrix2( aPathwayProxy ):
    '''
    calculate and return the elasticities (matrix)
    aPathwayProxy: a PathwayProxy instance
    return elasticityMatrix
    '''

    variableList = aPathwayProxy.getVariableList()
    processList = aPathwayProxy.getProcessList()

    elasticityMatrix = numpy.zeros( ( len( variableList ), len( processList ) ), numpy.Float )

    relationMatrix = aPathwayProxy.getRelationMatrix()
    independentGroupList = createIndependentGroupList( relationMatrix )
    
    activityBuffer = numpy.zeros( len( processList ), numpy.Float )

    aSession = aPathwayProxy.theEmlSupport.createInstance()

    aSession.step()
    for i in range( len( processList ) ):
        activityBuffer[ i ] = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )
    
    for groupList in independentGroupList:

        aSession = aPathwayProxy.theEmlSupport.createInstance()

        perturbationList = []
        for i in groupList:
            fullPN = variableList[ i ] + ':Value'
            value = aSession.theSimulator.getEntityProperty( fullPN )
            aPerturbation = RELATIVE_PERTURBATION * value + ABSOLUTE_PERTURBATION
            perturbationList.append( aPerturbation )
            aSession.theSimulator.setEntityProperty( fullPN, value + aPerturbation )

        aSession.step()

        for c in range( len( groupList ) ):
            i = groupList[ c ]
            aPerturbation = perturbationList[ c ]
            for j in range( len( processList ) ):
                if relationMatrix[ i ][ j ]:
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
        aPathwayProxy = anEmlSupport.createPathwayProxy()

        if fullPN != None:

            print 'elasticity array for \'%s\' =' % ( fullPN )
            print getElasticityArray( aPathwayProxy, fullPN )
            print 'acculate elasticity array for \'%s\' =' % ( fullPN )
            print getAcculateElasticityArray( aPathwayProxy, fullPN )
            print 'scaled elasticity array for \'%s\' =' % ( fullPN )
            print getScaledElasticity( aPathwayProxy, fullPN )

        print 'epsilon elasticity matrix ='
        print getEpsilonElasticityMatrix( aPathwayProxy )
        print 'scaled epsilon elasticity matrix ='
        print getScaledEpsilonElasticityMatrix( aPathwayProxy )
        print 'epsilon elasticity matrix ='
        print getEpsilonElasticityMatrix2( aPathwayProxy )

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/sample/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ), 'Variable:/CELL/CYTOPLASM:A13P2G:Value' )
