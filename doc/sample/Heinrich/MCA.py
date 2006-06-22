
from ecell.analysis.MatrixIO import writeMatrix
from ecell.analysis.emlsupport import EmlSupport

import numpy

import sys


if len( sys.argv ) > 1:
    anEmlSupport = EmlSupport( sys.argv[ 1 ] )
else:
    anEmlSupport = EmlSupport( './Heinrich.eml' )

aPathwayProxy = anEmlSupport.createPathwayProxy()

variableList = aPathwayProxy.getVariableList()
processList = aPathwayProxy.getProcessList()

# stoichiometry matrix

stoichiometryMatrix = aPathwayProxy.getStoichiometryMatrix()
writeMatrix( 'StoichiometryMatrix.csv', stoichiometryMatrix, variableList, processList )

# decomposed matrices

from ecell.analysis.Structure import generateFullRankMatrix

( linkMatrix, kernelMatrix, independentList ) = generateFullRankMatrix( stoichiometryMatrix )

reducedMatrix = numpy.take( stoichiometryMatrix, independentList )

reducedVariableList = []
for i in independentList:
    reducedVariableList.append( variableList[ i ] )

# elementary flux mode

from ecell.analysis.Structure import generateElementaryFluxMode

modeList = generateElementaryFluxMode( stoichiometryMatrix, aPathwayProxy.getReversibilityList() )
modeMatrix = numpy.transpose( numpy.array( modeList ) )
writeMatrix( 'ElementaryFluxMode.csv', modeMatrix, processList )

# elasticity matrix

from ecell.analysis.Elasticity import getEpsilonElasticityMatrix2

elasticityMatrix = getEpsilonElasticityMatrix2( aPathwayProxy )
writeMatrix( 'UnscaledElasticityMatrix.csv', elasticityMatrix, variableList, processList )

# Jacobian matrix

from ecell.analysis.Jacobian import getJacobianMatrix2

aJacobianMatrix = getJacobianMatrix2( aPathwayProxy )
writeMatrix( 'JacobianMatrix.csv', aJacobianMatrix, variableList, variableList )

# unscaled control coefficients

from ecell.analysis.ControlCoefficient import calculateUnscaledControlCoefficient

( unscaledCCCMatrix, unscaledFCCMatrix ) = calculateUnscaledControlCoefficient( stoichiometryMatrix, elasticityMatrix )

writeMatrix( 'UnscaledCCCMatrix.csv', unscaledCCCMatrix, variableList, processList )
writeMatrix( 'UnscaledFCCMatrix.csv', unscaledFCCMatrix, processList, processList )

# scaled control coefficients

from ecell.analysis.ControlCoefficient import scaleControlCoefficient

( scaledCCCMatrix, scaledFCCMatrix ) = scaleControlCoefficient( aPathwayProxy, unscaledCCCMatrix, unscaledFCCMatrix )

writeMatrix( 'ScaledCCCMatrix.csv', scaledCCCMatrix, variableList, processList )
writeMatrix( 'ScaledFCCMatrix.csv', scaledFCCMatrix, processList, processList )
