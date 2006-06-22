
"""
A program for calculating Jacobian
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'Jacobian'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


from util import RELATIVE_PERTURBATION, ABSOLUTE_PERTURBATION, allzero, createIndependentGroupList

import numpy


def getJacobianMatrix( aPathwayProxy ):
    '''
    calculate and return the Jacobian matrix (array)
    aPathwayProxy: a PathwayProxy instance
    return aJacobian
    '''

    variableList = aPathwayProxy.getVariableList()

    size = len( variableList )

    aJacobianMatrix = numpy.zeros( ( size, size ), numpy.Float )

    velocityBuffer = numpy.zeros( size, numpy.Float )

    aSession = aPathwayProxy.theEmlSupport.createInstance()
    aSession.step()
    
    for i in range( size ):
        velocityBuffer[ i ] = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Velocity' )

    # calculate derivatives
    for i in range( size ):

        aSession = aPathwayProxy.theEmlSupport.createInstance()

        value = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Value' )
        aPerturbation = value * RELATIVE_PERTURBATION + ABSOLUTE_PERTURBATION
        aSession.theSimulator.setEntityProperty( variableList[ i ] + ':Value', value + aPerturbation )

        aSession.step()

        for j in range( size ):
            aJacobianMatrix[ j ][ i ] = ( aSession.theSimulator.getEntityProperty( variableList[ j ] + ':Velocity' ) - velocityBuffer[ j ] ) / aPerturbation 

    return aJacobianMatrix

# end of getJacobianMatrix


def getJacobianMatrix2( aPathwayProxy ):
    '''
    calculate and return the Jacobian matrix (array)
    aPathwayProxy: a PathwayProxy instance
    return aJacobian
    '''

    variableList = aPathwayProxy.getVariableList()

    size = len( variableList )

    aJacobianMatrix = numpy.zeros( ( size, size ), numpy.Float )

    relationMatrix = numpy.matrixmultiply( aPathwayProxy.getRelationMatrix(), numpy.transpose( aPathwayProxy.getRelationMatrix( 1 ) ) )
    independentGroupList = createIndependentGroupList( relationMatrix )

    velocityBuffer = numpy.zeros( size, numpy.Float )

    aSession = aPathwayProxy.theEmlSupport.createInstance()

    aSession.step()
    for i in range( size ):
        velocityBuffer[ i ] = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Velocity' )

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
            for j in range( len( variableList ) ):
                if relationMatrix[ i ][ j ] != 0:
                    aJacobianMatrix[ j ][ i ] = ( aSession.theSimulator.getEntityProperty( variableList[ j ] + ':Velocity' ) - velocityBuffer[ j ] ) / aPerturbation

    return aJacobianMatrix

# end of getJacobianMatrix2


if __name__ == '__main__':

    from emlsupport import EmlSupport
    from Jacobian import *

    import sys
    import os


    def main( filename ):
        
        anEmlSupport = EmlSupport( filename )
        aPathwayProxy = anEmlSupport.createPathwayProxy()

        print 'Jacobian matrix ='
        print getJacobianMatrix( aPathwayProxy )
        print 'Jacobian matrix ='
        print getJacobianMatrix2( aPathwayProxy )

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/sample/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
