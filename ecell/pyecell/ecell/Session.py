#! /usr/bin/env python
import string
import eml
        
class Session:

    def __init__( self, aSimulator ):

        self.thePrintMethod = self.plainPrintMethod
        self.theSimulator = aSimulator

    def run( self , time='' ):
        if not time:
            self.theSimulator.run()
        else:
            self.theSimulator.run( time )

    def stop( self ):
        self.theSimulator.stop()

    def step( self, num='' ):
        if not num:
            self.theSimulator.step()
        else:
            for i in range(num):
                self.theSimulator.step()

    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()

    def getLogger( self, fullpn ):
        return self.theSimulator.getLogger( fullpn )

    def setPendingEventChecker( self, event ):

        self.theSimulator.setPendingEventChecker( event )

    def setEventHandler( self, event ):

        self.theSimulator.setEventHandler( event )

    def getCurrentTime( self ):

        return self.theSimulator.getCurrentTime()

    def setPrintMethod( self, aMethod ):
        self.thePrintMethod = aMethod

    def printMessage( self, message ):
        self.thePrintMethod( message )

    def loadModel( self, aFileObject ):
        anEmlParser = eml.EmlParser( aFileObject )
        self.__thePreModel = anEmlParser.parse()
        
        #self.__thePreModel = aModel

        self.loadStepper()
        self.loadEntity()
        self.loadSystemStepper()
        self.loadProperty()

        
    def saveModel( self ):
        pass

    def plainPrintMethod( aMessage ):
        print aMessage





    def loadStepper( self ):
        """stepper loader"""

        for aTargetStepper in( self.__thePreModel[ 'stepper' ] ):
            aClass = aTargetStepper[0]
            anId   = aTargetStepper[1]

            ## TemporarySample ---------------------------------------
            #aPrintClass = "self.theSimulator.createStepper('" + aClass + "',"
            #aPrintId     = "'" + anId + "')"
            #print aPrintClass, aPrintId ## Temporary
            ## -------------------------------------------------------

            self.theSimulator.createStepper( aClass, anId )




    def loadEntity( self ):
        """Entity loader"""

        for aTargetEntity in( self.__thePreModel[ 'entity' ] ):
            aType = aTargetEntity[0]
            aPath = aTargetEntity[1]
            anId  = aTargetEntity[2]
            aName = aTargetEntity[3]


            ## need refactoring about PathConvert!!                
            if aType == 'System':
                if aPath == '/':
                    aPath = ''
                else:
                    aPathList = aPath.split( '/' )
                    aLenPathList = len( aPathList )
                    aPathList = aPathList[0:aLenPathList-1]
                    aPath     = string.join( aPathList, '/' )
                    if aPath == '':
                        aPath = '/'
            ## --------------------------------------------------

            if not ( aType == 'System' or aType == 'Substance' ):
                anEntityType = 'Reactor'
                aFullId = anEntityType + ':' + aPath + ':' + anId
            else:
                aFullId = aType + ':' + aPath + ':' + anId



            ## check!!
            if not aPath == '':
                self.theSimulator.createEntity( aType, aFullId, aName )

                ##Temporary
                #print 'self.theSimulator.createEntity(', aType, ',', aFullId, ',', aName, ')'
            


    def loadSystemStepper( self ):
        
        for aTargetStepperSystem in( self.__thePreModel[ 'stepper_system' ] ):

            aFullPath = aTargetStepperSystem[0]
            aFullPathList = aFullPath.split( '/' )

            aClass  = aTargetStepperSystem[1]
            aValueList = [ aClass ]
            aName = 'StepperID'


            if aFullPath == '/':
                aFullPn = 'System::/:' + aName
                
            else:
                anId    = aFullPathList[-1]
                aFullPathList.remove( aFullPathList[ len(aFullPathList) - 1 ] )
                aPath = string.join( aFullPathList, '/' )
                if aPath == '':
                    aPath = '/'
                aFullPn = 'System' + ':' + aPath + ':' + anId + ':' + aName

            self.theSimulator.setProperty( aFullPn, aValueList )

            ##Temporary
            #print 'self.theSimulator.setProperty(', aFullPn, ',', aValueList, ')'
            



            

    def loadProperty( self ):
        """Property loader"""
        
        for aTargetProperty in( self.__thePreModel[ 'property' ] ):
            aFullPn = aTargetProperty[0]
            aValue  = aTargetProperty[1]

            
            self.theSimulator.setProperty( aFullPn, aValue )

            ## TemporarySample ---------------------------------
            #aPrintFullPn = "self.theSimulator.setProperty('" + aFullPn + "',"
            #print aPrintFullPn, aValue, ')'
            ## -------------------------------------------------






