from ecell.ECDDataFile import *
import os
import ecell
import ecell.emc
import ecell.Session
import ecell.ecs
                
def readTable( filename ):

    fileObj = open( filename, 'r' )
    
    line = fileObj.readline().rstrip()
    fullPNList = [ eval( x ) for x in line.split( "," )[1:] ]

    line = fileObj.readline().rstrip()
    weightList = [ eval( x ) for x in line.split( "," )[1:] ]
    totalWeight = float( sum( weightList ) )
    weightList = [ x / totalWeight for x in weightList ]

    dataList = []
    while ( 1 ):
        line = fileObj.readline()
	if line == "":
	    break
        data = [ eval( x ) for x in line.rstrip().split( "," ) ]
	dataList.append( data )

    return ( fullPNList, weightList, dataList )

# readTable

_DATAcsv_ = 'answer.csv' 

## When you debug this file, delete forehand # of the following
## parameters and execute,
## [%] ecell3-session runsession.py

#### comment out region ####
# _DATA_ = 'Data'
# _K_ = -45

# model script

sim = theSimulator.createStepper('SpatiocyteStepper', 'SS').VoxelRadius = 6e-9
sim2 = theSimulator.createStepper('ODEStepper', 'DE')

theSimulator.rootSystem.StepperID = 'SS'
theSimulator.createEntity('Variable', 'Variable:/:GEOMETRY').Value = 5
theSimulator.createEntity('Variable', 'Variable:/:LENGTHX').Value = 1e-6
theSimulator.createEntity('Variable', 'Variable:/:LENGTHY').Value = 1e-6
theSimulator.createEntity('Variable', 'Variable:/:LENGTHZ').Value = 1e-6
theSimulator.createEntity('Variable', 'Variable:/:VACANT')

A = theSimulator.createEntity('Variable', 'Variable:/:A')
A.Value = 10000
A.Name = "HD"
B = theSimulator.createEntity('Variable', 'Variable:/:B')
B.Value = 10000
B.Name = "HD"
C = theSimulator.createEntity('Variable', 'Variable:/:C')
C.Value = 0
C.Name = "HD"
Am = theSimulator.createEntity('Variable', 'Variable:/:Am')
Am.Value = 10000
Am.Name = "HD"
Bm = theSimulator.createEntity('Variable', 'Variable:/:Bm')
Bm.Value = 10000
Bm.Name = "HD"
Cm = theSimulator.createEntity('Variable', 'Variable:/:Cm')
Cm.Value = 0
Cm.Name = "HD"

binder = theSimulator.createEntity('SpatiocyteNextReactionProcess', 'Process:/:reaction')
binder.VariableReferenceList = [['_', 'Variable:.:A', '-2'], ['_', 'Variable:.:B', '-1'], ['_', 'Variable:.:C', '1']]
binder.k = pow(10, _K_ )

binder = theSimulator.createEntity('MassActionProcess', 'Process:/:reaction2')
binder.StepperID = 'DE'
binder.VariableReferenceList = [['_', 'Variable:.:Am', '-2'], ['_', 'Variable:.:Bm', '-1'], ['_', 'Variable:.:Cm', '1']]
binder.k = pow(10, _K_ )

# run model

( fullPNList, weightList, dataList ) =  readTable( _DATA_ + os.sep + _DATAcsv_ )

initDict = {}
for fullPNString in fullPNList:
    initDict[ fullPNString ] = theSimulator.getEntityProperty( fullPNString )
    print fullPNString, initDict[ fullPNString ]

currentTime = 0.0
totalDifference = 0.0

for data in dataList[1:]:
    nextTime = data[ 0 ]
    if nextTime > currentTime:
	run( nextTime - currentTime )
	currentTime = nextTime

    print currentTime
    
    aDifference = 0.0
    for ( fullPNString, testValue, initValue, weight ) \
            in zip( fullPNList, data[1:], dataList[ 0 ][1:], weightList ):
        aValue = theSimulator.getEntityProperty( fullPNString )
        evaluatedValue = testValue
        aDifference += weight * ( ( evaluatedValue - aValue ) / evaluatedValue ) * ( ( evaluatedValue - aValue ) / evaluatedValue )
        print fullPNString, aValue

    totalDifference += aDifference

totalDifference /= len( dataList ) - 1
print totalDifference

open( 'result.dat', 'w' ).write( str( totalDifference ) )

