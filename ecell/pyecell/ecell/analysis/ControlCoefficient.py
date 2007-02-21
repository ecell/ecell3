
"""
A program for calculating control coefficients
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'ControlCoefficient'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


from Structure import generateFullRankMatrix

import numpy
import numpy.lib.twodim_base
import numpy.linalg


from Elasticity import getEpsilonElasticityMatrix2
GET_ELASTICITY_MATRIX = getEpsilonElasticityMatrix2


def calculateControlCoefficient( aPathwayProxy, mode=0 ):
    '''
    calculate concentration and flux control coefficients
    aPathwayProxy: a PathwayProxy instance
    mode: (0 or 1) unscaled or scaled
    return ( CCCMatrix, FCCMatrix )
    '''

    stoichiometryMatrix = aPathwayProxy.getStoichiometryMatrix()
    elasticityMatrix = GET_ELASTICITY_MATRIX( aPathwayProxy )

    ( unscaledCCCMatrix, unscaledFCCMatrix ) = calculateUnscaledControlCoefficient( stoichiometryMatrix, elasticityMatrix )

    if not mode:
        return ( unscaledCCCMatrix, unscaledFCCMatrix )
    else:
        return scaleControlCoefficient( aPathwayProxy, \
                                        unscaledCCCMatrix, unscaledFCCMatrix )

# end of calculateControlCoefficient


def calculateUnscaledControlCoefficient( stoichiometryMatrix, elasticityMatrix ):
    '''
    calculate unscaled concentration and flux control coefficients from stoichiometry and elasticity matrices
    stoichometryMatrix: (matrix)
    elasticityMatrix: (matrix)
    return ( unscaledCCCMatrix, unscaledFCCMatrix )
    '''

    ( m, n ) = numpy.shape( stoichiometryMatrix )

    # do Gaussian elimination,
    # and get reduced stoichiometry, kernel and link matrix
    # see util.generateFullRankMatrix for the details
    ( linkMatrix, kernelMatrix, independentList ) = generateFullRankMatrix( stoichiometryMatrix )
    reducedMatrix = numpy.take( stoichiometryMatrix, independentList )

    # constract Jacobian matrix from reduced, link matrix and elasticities,
    # M0 = N0 * epsilon * L
    epsilonLMatrix = numpy.matrixmultiply( numpy.transpose( elasticityMatrix ), linkMatrix )
    aJacobianMatrix = numpy.matrixmultiply( reducedMatrix, epsilonLMatrix )

    # calculate unscaled concentration control coefficients
    # CS = -L * (M0)^(-1) * N0
    invJacobian = numpy.linalg.inv( aJacobianMatrix )

    unscaledCCCMatrix = numpy.matrixmultiply( -1.0 * linkMatrix, invJacobian )
    unscaledCCCMatrix = numpy.matrixmultiply( unscaledCCCMatrix, reducedMatrix )

    # calculate unscaled flux control coefficients
    # CJ = I - epsilon * CS

    unscaledFCCMatrix = numpy.identity( n ) + numpy.matrixmultiply( numpy.transpose( elasticityMatrix ), unscaledCCCMatrix )

    return ( unscaledCCCMatrix, unscaledFCCMatrix )

# end of calcuteUnscaledControlCoefficient


def invdiag( traceArray ):
    '''
    return numpy.lib.twodim_base.diag( 1.0 / array )
    if there\'re zeros in the array, set zero for that
    traceArray: (array) one dimensional array
    return (matrix)
    '''

    size = len( traceArray )
    invArray = numpy.zeros( size, numpy.Float )
    for i in range( size ):
        if abs( traceArray[ i ] ) > 0.0:
            invArray[ i ] = 1.0 / traceArray[ i ]

    return numpy.lib.twodim_base.diag( invArray )

# end of invdiag


def scaleControlCoefficient( aPathwayProxy, unscaledCCCMatrix, unscaledFCCMatrix ):
    '''
    scale concentration and flux control coefficients and return scaled matrices
    aPathwayProxy: a PathwayProxy instance
    unscaledCCCMatrix: (matrix)
    unscaledFCCMatrix: (matrix)
    return ( unscaledCCCMatrix, unscaledFCCMatrix )
    '''

    # calculate initial activities and get initial values
    processList = aPathwayProxy.getProcessList()
    variableList = aPathwayProxy.getVariableList()
    
    aSession = aPathwayProxy.theEmlSupport.createInstance()

    aSession.step()
    
    activityBuffer = numpy.zeros( len( processList ), numpy.Float )
    for i in range( len( processList ) ):
        activityBuffer[ i ] = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )

    valueBuffer = numpy.zeros( len( variableList ), numpy.Float )
    for i in range( len( variableList ) ):
        valueBuffer[ i ] = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Value' )

    # calculate scaled concentration control coefficient
    # ( scaled CS_ij ) = E_j / S_i * ( unscaled CS_ij )

    scaledCCCMatrix = numpy.matrixmultiply( invdiag( valueBuffer ), unscaledCCCMatrix )
    scaledCCCMatrix = numpy.matrixmultiply( scaledCCCMatrix, numpy.lib.twodim_base.diag( activityBuffer ) )

    # calculate scaled flux control coefficient
    # ( scaled CJ_ij ) = E_j / E_i * ( unscaled CJ_ij )

    scaledFCCMatrix = numpy.matrixmultiply( invdiag( activityBuffer ), unscaledFCCMatrix )
    scaledFCCMatrix = numpy.matrixmultiply( scaledFCCMatrix, numpy.lib.twodim_base.diag( activityBuffer ) )

    return ( scaledCCCMatrix, scaledFCCMatrix )

# end of scaleControlCoefficient


if __name__ == '__main__':

    from emlsupport import EmlSupport

    import sys
    import os


    def main( filename ):
        
        anEmlSupport = EmlSupport( filename )
        aPathwayProxy = anEmlSupport.createPathwayProxy()

        ( unscaledCCCMatrix, unscaledFCCMatrix ) = calculateControlCoefficient( aPathwayProxy )

        print 'unscaled concentration control coefficients ='
        print unscaledCCCMatrix
        print 'unscaled flux control coefficients ='
        print unscaledFCCMatrix

        ( scaledCCCMatrix, scaledFCCMatrix ) = scaleControlCoefficient( aPathwayProxy, unscaledCCCMatrix, unscaledFCCMatrix )
        # ( scaledCCCMatrix, scaledFCCMatrix ) = calculateControlCoefficient( aPathwayProxy, 1 )

        print 'scaled concentration control coefficients ='
        print scaledCCCMatrix
        print 'scaled flux control coefficients ='
        print scaledFCCMatrix

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/sample/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
