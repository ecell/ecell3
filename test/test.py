import ecell.Session
import ecell.ecs


from ecell.FullID import *

from ecell.util import *
from ecell.ECS import *

from Numeric import *

print 'create a Session'
aSession = ecell.Session.Session( ecell.ecs.Simulator() )
aSimulator = aSession.theSimulator

aSimulator.createStepper( 'RungeKutta4SRMStepper', 'RK4_0' )
aSimulator.createStepper( 'RungeKutta4SRMStepper', 'RK4_1' )
aSimulator.createStepper( 'Euler1SRMStepper', 'E1_0' )

aSimulator.setProperty( 'System::/:StepperID', ('RK4_0',) )

print aSimulator.getStepperProperty( 'RK4_0', 'StepInterval' )
print aSimulator.getStepperProperty( 'RK4_0', 'SystemList' )

print 'make variables...'
aSimulator.createEntity( 'SRMVariable', 'Variable:/:A', 'variable A' )
aSimulator.createEntity( 'SRMVariable', 'Variable:/:B', 'variable B' )

aSimulator.run(10)

print 'initialize()...'
aSimulator.initialize()

aSimulator.run(10)

aSimulator.createEntity( 'System', 'System:/:CYTOPLASM', 'cytoplasm' )
aSimulator.setProperty( 'System:/:CYTOPLASM:StepperID', ('RK4_0',) )


aSimulator.createEntity( 'SRMVariable', 'Variable:/CYTOPLASM:CA', 's CA' )
aSimulator.createEntity( 'SRMVariable', 'Variable:/CYTOPLASM:CB', 's CB' )


print 'initialize()...'
aSimulator.initialize()


print 'set Variable:/:A Value = 30'
aSimulator.setProperty( 'Variable:/:A:Value', (30,) )


printAllProperties( aSimulator, 'System::/' )

printAllProperties( aSimulator, 'System:/:CYTOPLASM' )


variablelist = aSimulator.getProperty( 'System::/:VariableList' )

for i in variablelist:
    printAllProperties( aSimulator, 'Variable:/:' + i )

variablelist = aSimulator.getProperty( 'System:/:CYTOPLASM:VariableList' )

for i in variablelist:
    printAllProperties( aSimulator, 'Variable:/CYTOPLASM:' + i )

print

printProperty( aSimulator, 'Variable:/:A:Value' )

#printProperty( aSimulator, 'Variable:/:A:Value' )
print 'changing Value of Variable:/:A...'
aSimulator.setProperty( 'Variable:/:A:Value', (10.0, ) )
printProperty( aSimulator, 'Variable:/:A:Value' )


print 'step()...'
print aSimulator.getCurrentTime()
aSimulator.step()

print aSimulator.getCurrentTime()
aSimulator.step()

print aSimulator.getCurrentTime()
aSimulator.step()

print aSimulator.getCurrentTime()
aSimulator.step()

print "getLogger"
aALogger = aSimulator.getLogger('Variable:/:A:Value')
aBLogger = aSimulator.getLogger('Variable:/:B:Value')

print aSimulator.getCurrentTime()
aSimulator.run(10)
print aSimulator.getCurrentTime()

start= aALogger.getStartTime()



    

end  = aALogger.getEndTime()
print start ,end
print aSimulator.getLoggerList()

############################################
#setResultFileList()#

a = array(([20,30],
           [40,50],
           [60,70],
           [80,90],
           [100,110],
           [120,130],
           [140,150],
           [160,170],
           [180,190],
           [200,210],
           [220,230],
           [240,250],
           [260,270],
           [280,290],
           [300,310],
           [320,330],
           [340,350]))  #aALogger.getData()

sizeOfArrayInArray1 = len(a[0])
sizeOfArrayInArray2 = len(a[1])
sizeOfArray = len(a)

List = list( aSimulator.getLoggerList() )

for x in List: 
    tmpx = string.replace(x, ':', '-')
    string.replace(tmpx, '/', '_')

appendedList = List.append(tmpx)
print appendedList
    
data = 'DATA:' + str( List )
size = "SIZE: %s %s"%(sizeOfArrayInArray1, sizeOfArray)
label = "LABEL: time value"
note = "NOTE:\n\n-------------------------"

stringList = [ data , size , label , note ]
numericList = [sizeOfArrayInArray1 , sizeOfArrayInArray2 ]

output = open('test.test','w')
for x in stringList:
    print x
    output.writelines("%s\n"%x)
    
    for numericList in a:
        print numericList
        tmp = ''
        for y in numericList:
            
            tmp = tmp + "%10s"%y 

        output.writelines(tmp+"\n")
             
    output.writelines("\n///")        
            
            
#if  __name__ == "__main__":
#    
#    setResultFileList("result.ecd")


############################################
# added for logger
print "\n\nstart to save log "

# import library ECD file format 
from ecell.ECDDataFile import *

#print List

# save data 
# List is the FullPN list which has created by getLogger() method.
# In this sample, List include three FullPNs,
# Saving [Variable:/:A:Value] will be success.
# Saving [Variable:/:B:Value] will be success.
# Saving [Variable-/-B-Value] will be failure.

for aFullPNString in List:
    try:
        #print aFullPNString

        # creates save file name
        # ex.  [Variable:/:A:Value] -> [A_Value]
        aFileName=join( split(aFullPNString,':')[2:], '_' )
        #print " aFileName = %s" %aFileName 

        # ----------------------------------------------------
        # If you want to specify the start time and end time, 
        # uncomment out following lines.
        #aStartTime = aSimulator.getLogger(aFullPNString).getStartTime()
        #aEndTime = aSimulator.getLogger(aFullPNString).getEndTime()
        #aMatrixData = aSimulator.getLogger(aFullPNString).getData( aStartTime, aEndTime )

        # ----------------------------------------------------
        # If you want to specify the start time, end time and 
        # interval time, uncomment out following lines.
        #aStartTime = aSimulator.getLogger(aFullPNString).getStartTime()
        #aEndTime = aSimulator.getLogger(aFullPNString).getEndTime()
        #anInterval = 10
        #aMatrixData = aSimulator.getLogger(aFullPNString).getData( aStartTime, aEndTime, anInterval )

        # ----------------------------------------------------
        # If you want to get default dat, uncomment out followint line.
        aMatrixData = aSimulator.getLogger(aFullPNString).getData()

        # creates instance of ECD format file
        anECDFile = ECDDataFile()

        # sets some properties to ECD format file instance
        anECDFile.setDataName(aFileName)
        anECDFile.setLabel('test_data_label')
        anECDFile.setNote('test_data_note')
        anECDFile.setMatrixData( aMatrixData )
        anECDFile.setFileName( aFileName )

        # excutes saving file
        anECDFile.save()

    except:
        print 'couldn\'t save [%s] data to [%s]' %(aFullPNString,aFileName)
    else:
        print 'saved [%s] data to [%s]' %(aFullPNString,aFileName)

print "end of saving log "


