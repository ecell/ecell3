#! /usr/bin/env python
import string
        
class Session:

    def __init__( self, aSimulator ):

        self.thePrintMethod = self.plainPrintMethod
        self.theSimulator = aSimulator

    def run( self , time='' ):
        if not time:
            self.theSimulator.run()
        else:
            self.theSimulator.run( time )

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

    def loadModel( self, aModel ):
        self.__thePreModel = aModel

        self.loadEntity()
        self.loadStepper()
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
            
            ## TemporarySample ---------------------------------------
            #aPrintFullId = "self.theSimulator.setProperty('" + aFullPn + "',"
            #print aPrintFullId, aValueList, ")"
            ##-------------------------------------------------------

            


    def loadEntity( self ):
        """Entity loader"""

        for aTargetEntity in( self.__thePreModel[ 'entity' ] ):
            aType = aTargetEntity[0]
            aPath = aTargetEntity[1]
            anId  = aTargetEntity[2]
            aName = aTargetEntity[3]

            if aType == 'System':

                ## need refactoring about PathConvert!!                
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

                    aFullId = aType + ':' + aPath + ':' + anId
                    self.theSimulator.createEntity( aType, aFullId, aName )

            ## TemporarySample ---------------------------------------
            #aPrintFullId = "self.theSimulator.createEntity('" + aType + "',"
            #aPrintType   = "'" + aFullId + "',"
            #aPrintName   = "'" + aName + "')"
            #print aPrintFullId, aPrintType, aPrintName ## Temporary
            ## -------------------------------------------------------


            

    def loadProperty( self ):
        """Property loader"""
        
        for aTargetProperty in( self.__thePreModel[ 'property' ] ):
            aFullPn = aTargetProperty[0]
            aValue  = aTargetProperty[1]

            
            ## TemporarySample ---------------------------------
            #aPrintFullPn = "self.theSimulator.setProperty('" + aFullPn + "',"
            #print aPrintFullPn, aValue, ')'
            ## -------------------------------------------------

            self.theSimulator.setProperty( aFullPn, aValue )






