import ecell.Session
import ecell.ecs


from ecell.FullID import *

from ecell.util import *
from ecell.ECS import *

from Numeric import *

print 'create a Session'
aSession = ecell.Session.Session( ecell.ecs.Simulator() )
aSimulator = aSession.theSimulator

aSimulator.createStepper( 'RungeKutta4SRMStepper', 'RK4_0', () )
aSimulator.createStepper( 'RungeKutta4SRMStepper', 'RK4_1', () )
aSimulator.createStepper( 'Euler1SRMStepper', 'E1_0', () )

aSimulator.setProperty( 'System::/:StepperID', ('RK4_0',) )



print 'make substances...'
aSimulator.createEntity( 'SRMSubstance', 'Substance:/:A', 'substance A' )
aSimulator.createEntity( 'SRMSubstance', 'Substance:/:B', 'substance B' )

aSimulator.run(10)

print 'initialize()...'
aSimulator.initialize()

aSimulator.run(10)

aSimulator.createEntity( 'System', 'System:/:CYTOPLASM', 'cytoplasm' )
aSimulator.setProperty( 'System:/:CYTOPLASM:StepperID', ('RK4_0',) )


aSimulator.createEntity( 'SRMSubstance', 'Substance:/CYTOPLASM:CA', 's CA' )
aSimulator.createEntity( 'SRMSubstance', 'Substance:/CYTOPLASM:CB', 's CB' )


print 'initialize()...'
aSimulator.initialize()


print 'set Substance:/:A Quantity = 30'
aSimulator.setProperty( 'Substance:/:A:Quantity', (30,) )


printAllProperties( aSimulator, 'System::/' )

printAllProperties( aSimulator, 'System:/:CYTOPLASM' )


substancelist = aSimulator.getProperty( 'System::/:SubstanceList' )

for i in substancelist:
    printAllProperties( aSimulator, 'Substance:/:' + i )

substancelist = aSimulator.getProperty( 'System:/:CYTOPLASM:SubstanceList' )

for i in substancelist:
    printAllProperties( aSimulator, 'Substance:/CYTOPLASM:' + i )

print

printProperty( aSimulator, 'Substance:/:A:Quantity' )

#printProperty( aSimulator, 'Substance:/:A:Quantity' )
print 'changing Quantity of Substance:/:A...'
aSimulator.setProperty( 'Substance:/:A:Quantity', (10.0, ) )
printProperty( aSimulator, 'Substance:/:A:Quantity' )


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
aALogger = aSimulator.getLogger('Substance:/:A:Quantity')
aBLogger = aSimulator.getLogger('Substance:/:B:Quantity')

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
label = "LABEL: time quantity"
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
# Saving [Substance:/:A:Quantity] will be success.
# Saving [Substance:/:B:Quantity] will be success.
# Saving [Substance-/-B-Quantity] will be failure.

for aFullPNString in List:
    try:
        #print aFullPNString

        # creates save file name
        # ex.  [Substance:/:A:Quantity] -> [A_Quantity]
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


