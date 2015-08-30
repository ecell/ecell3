#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2015 Keio University
#       Copyright (C) 2008-2015 RIKEN
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


def calculateControlCoefficient( pathwayProxy, mode=0 ):
    '''
    calculate concentration and flux control coefficients
    pathwayProxy: a PathwayProxy instance
    mode: (0 or 1) unscaled or scaled
    return ( CCCMatrix, FCCMatrix )
    '''

    stoichiometryMatrix = pathwayProxy.getStoichiometryMatrix()
    elasticityMatrix = GET_ELASTICITY_MATRIX( pathwayProxy )

    ( unscaledCCCMatrix, unscaledFCCMatrix ) = calculateUnscaledControlCoefficient( stoichiometryMatrix, elasticityMatrix )

    if not mode:
        return ( unscaledCCCMatrix, unscaledFCCMatrix )
    else:
        return scaleControlCoefficient( pathwayProxy, \
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
    reducedMatrix = numpy.take( stoichiometryMatrix, independentList, 0 )

    # constract Jacobian matrix from reduced, link matrix and elasticities,
    # M0 = N0 * epsilon * L
    epsilonLMatrix = numpy.dot( numpy.transpose( elasticityMatrix ), linkMatrix )
    aJacobianMatrix = numpy.dot( reducedMatrix, epsilonLMatrix )

    # calculate unscaled concentration control coefficients
    # CS = -L * (M0)^(-1) * N0
    invJacobian = numpy.linalg.inv( aJacobianMatrix )

    unscaledCCCMatrix = numpy.dot( -1.0 * linkMatrix, invJacobian )
    unscaledCCCMatrix = numpy.dot( unscaledCCCMatrix, reducedMatrix )

    # calculate unscaled flux control coefficients
    # CJ = I - epsilon * CS

    unscaledFCCMatrix = numpy.identity( n ) + numpy.dot( numpy.transpose( elasticityMatrix ), unscaledCCCMatrix )

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
    invArray = numpy.zeros( size, float )
    for i in range( size ):
        if abs( traceArray[ i ] ) > 0.0:
            invArray[ i ] = 1.0 / traceArray[ i ]

    return numpy.lib.twodim_base.diag( invArray )

# end of invdiag


def scaleControlCoefficient( pathwayProxy, unscaledCCCMatrix, unscaledFCCMatrix ):
    '''
    scale concentration and flux control coefficients and return scaled matrices
    pathwayProxy: a PathwayProxy instance
    unscaledCCCMatrix: (matrix)
    unscaledFCCMatrix: (matrix)
    return ( unscaledCCCMatrix, unscaledFCCMatrix )
    '''

    # calculate initial activities and get initial values
    processList = pathwayProxy.getProcessList()
    variableList = pathwayProxy.getVariableList()
    
    aSession = pathwayProxy.theEmlSupport.createSession()

    aSession.step()
    
    activityBuffer = numpy.zeros( len( processList ), float )
    for i in range( len( processList ) ):
        activityBuffer[ i ] = aSession.theSimulator.getEntityProperty( processList[ i ] + ':Activity' )

    valueBuffer = numpy.zeros( len( variableList ), float )
    for i in range( len( variableList ) ):
        valueBuffer[ i ] = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Value' )

    # calculate scaled concentration control coefficient
    # ( scaled CS_ij ) = E_j / S_i * ( unscaled CS_ij )

    scaledCCCMatrix = numpy.dot( invdiag( valueBuffer ), unscaledCCCMatrix )
    scaledCCCMatrix = numpy.dot( scaledCCCMatrix, numpy.lib.twodim_base.diag( activityBuffer ) )

    # calculate scaled flux control coefficient
    # ( scaled CJ_ij ) = E_j / E_i * ( unscaled CJ_ij )

    scaledFCCMatrix = numpy.dot( invdiag( activityBuffer ), unscaledFCCMatrix )
    scaledFCCMatrix = numpy.dot( scaledFCCMatrix, numpy.lib.twodim_base.diag( activityBuffer ) )

    return ( scaledCCCMatrix, scaledFCCMatrix )

# end of scaleControlCoefficient


if __name__ == '__main__':

    from emlsupport import EmlSupport

    import sys
    import os


    def main( filename ):
        
        anEmlSupport = EmlSupport( filename )
        pathwayProxy = anEmlSupport.createPathwayProxy()

        ( unscaledCCCMatrix, unscaledFCCMatrix ) = calculateControlCoefficient( pathwayProxy )

        print 'unscaled concentration control coefficients ='
        print unscaledCCCMatrix
        print 'unscaled flux control coefficients ='
        print unscaledFCCMatrix

        ( scaledCCCMatrix, scaledFCCMatrix ) = scaleControlCoefficient( pathwayProxy, unscaledCCCMatrix, unscaledFCCMatrix )
        # ( scaledCCCMatrix, scaledFCCMatrix ) = calculateControlCoefficient( pathwayProxy, 1 )

        print 'scaled concentration control coefficients ='
        print scaledCCCMatrix
        print 'scaled flux control coefficients ='
        print scaledFCCMatrix

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
