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


def getJacobianMatrix( pathwayProxy ):
    '''
    calculate and return the Jacobian matrix (array)
    pathwayProxy: a PathwayProxy instance
    return aJacobian
    '''

    variableList = pathwayProxy.getVariableList()

    size = len( variableList )

    aJacobianMatrix = numpy.zeros( ( size, size ), float )

    velocityBuffer = numpy.zeros( size, float )

    aSession = pathwayProxy.theEmlSupport.createSession()
    aSession.theSimulator.initialize()
    
    for i in range( size ):
        velocityBuffer[ i ] = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Velocity' )

    # calculate derivatives
    for i in range( size ):

        aSession = pathwayProxy.theEmlSupport.createSession()

        value = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Value' )
        aPerturbation = value * RELATIVE_PERTURBATION + ABSOLUTE_PERTURBATION
        aSession.theSimulator.setEntityProperty( variableList[ i ] + ':Value', value + aPerturbation )

        aSession.theSimulator.initialize()

        for j in range( size ):
            aJacobianMatrix[ j ][ i ] = ( aSession.theSimulator.getEntityProperty( variableList[ j ] + ':Velocity' ) - velocityBuffer[ j ] ) / aPerturbation 

    return aJacobianMatrix

# end of getJacobianMatrix


def getJacobianMatrix2( pathwayProxy ):
    '''
    calculate and return the Jacobian matrix (array)
    pathwayProxy: a PathwayProxy instance
    return aJacobian
    '''

    variableList = pathwayProxy.getVariableList()

    size = len( variableList )

    aJacobianMatrix = numpy.zeros( ( size, size ), float )

    incidentMatrix = numpy.dot( pathwayProxy.getIncidentMatrix(), numpy.transpose( pathwayProxy.getIncidentMatrix( 1 ) ) )
    independentGroupList = createIndependentGroupList( incidentMatrix )

    velocityBuffer = numpy.zeros( size, float )

    aSession = pathwayProxy.theEmlSupport.createSession()

    aSession.theSimulator.initialize()
    for i in range( size ):
        velocityBuffer[ i ] = aSession.theSimulator.getEntityProperty( variableList[ i ] + ':Velocity' )

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
            for j in range( len( variableList ) ):
                if incidentMatrix[ i ][ j ] != 0:
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
        pathwayProxy = anEmlSupport.createPathwayProxy()

        print 'Jacobian matrix ='
        print getJacobianMatrix( pathwayProxy )
        print 'Jacobian matrix ='
        print getJacobianMatrix2( pathwayProxy )

        # import numpy.linalg
        # print numpy.linalg.inv( getJacobianMatrix( pathwayProxy ) )

    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
