from ModelEditor import *
from ModelStore import *
from BufferFactory import *
from Buffer import *
from Utils import *
from Constants import *
from DMInfo import *
from Command import *

class CreateStepper(ModelCommand):
    """
    arg0: Classname
    arg1: ID
    """
    CLASSNAME = 0
    ID = 1
    ARGS_NO = 2


    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.ID ]
        self.__theClassName = self.theArgs[ self.CLASSNAME ]
        if self.__theID in self.theModel.getStepperList():
            return False
        return True


    def do( self ):
        self.theModel.createStepper( self.__theClassName, self.__theID )
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = [ DeleteStepperList( self.theReceiver, [ self.__theID ] ) ]


    def getAffected( self ):
        return (ME_STEPPER_TYPE, None )


class DeleteStepperList(ModelCommand):
    """
    args:
    """
    IDLIST = 0
    ARGS_NO = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theIDList = self.theArgs[ self.IDLIST ]
        return True

    
    def do( self ):
        
        reverseBuffer = self.theBufferFactory.createStepperListBuffer( )
        for anID in self.__theIDList:
            tempBuffer = self.theBufferFactory.createStepperListBuffer( [anID] )
            self.theModel.deleteStepper( anID )
            self.theBufferFactory.incrementStepperBuffer( reverseBuffer, tempBuffer )
        
        self.theReverseCommandList = [ PasteStepperList( self.theReceiver, reverseBuffer ) ]
        return True

    def getAffected( self ):
        return (ME_STEPPER_TYPE, None )
                
            
    def createReverseCommand( self ):
        #reverse commandlist is created throughout do command
        self.theReverseCommandList = None
        



class RenameStepper(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    OLDID = 0
    NEWID = 1

    def checkArgs( self ):
        # oldID should exist
        # newID shouldn't exist
        if not ModelCommand.checkArgs(self):
            return False
        self.__theOldID = self.theArgs[ self.OLDID ]
        self.__theNewID = self.theArgs[ self.NEWID ]
        stepperList = self.theModel.getStepperList()
        if self.__theOldID not in stepperList:
            return False
        if self.__theNewID in stepperList:
            return False
        return True


    def do( self ):

        # store properties
        propBuffer = self.theBufferFactory.createStepperPropertyListBuffer( self.__theOldID, self.theModel.getStepperPropertyList( self.__theOldID ) )
        aClassName = self.theModel.getStepperClassName( self.__theOldID )

        # delete stepper
        self.theModel.deleteStepper( self.__theOldID )

        # create newstepper
        self.theModel.createStepper( aClassName, self.__theNewID )

        # paste property list
        self.theBufferPaster.pasteStepperPropertyListBuffer( propBuffer, self.__theNewID )

        # set system properties
        systemList = propBuffer.getProperty( ME_STEPPER_SYSTEMLIST )
        for aSystem in systemList:
            self.theModel.setEntityProperty( aSystem + ':' + 'StepperID',  self.__theNewID  )
        
        # set process properties
        processList = propBuffer.getProperty( ME_STEPPER_PROCESSLIST )
        for aProcess in processList:
            self.theModel.setEntityProperty( aProcess + ':' + 'StepperID',  self.__theNewID  )
        return True


    def createReverseCommand( self ):
        reversecommand = RenameStepper( self.theReceiver, self.__theNewID, self.__theOldID )
        self.theReverseCommandList = [ reversecommand ]

    def getAffected( self ):
        return (ME_STEPPER_TYPE, None )



class CopyStepperList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 1
    IDLIST = 0

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theIDList = self.theArgs[ self.IDLIST ]
        return True

    
    def do( self ):
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createStepperListBuffer( self.__theIDList ) )
        return True


    def createReverseCommand( self ):
        # there is no reverse command for copy
        self.theReverseCommandList = None

    def getAffected( self ):
        return (None, None )


class CutStepperList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 1
    IDLIST = 0

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theIDList = self.theArgs[ self.IDLIST ]
        return True

    
    def do( self ):
        # create copy buffer
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createStepperListBuffer( self.__theIDList ) )
        
        # delete list
        for anID in self.__theIDList:
            try:
                self.theModel.deleteStepper( anID )
            except Exception:
                pass
        return True


    def createReverseCommand( self ):
        reverseBuffer = self.theBufferFactory.createStepperListBuffer( self.__theIDList )
        self.theReverseCommandList = [ PasteStepperList( self.theReceiver, reverseBuffer ) ]
        
    def getAffected( self ):
        return (ME_STEPPER_TYPE, None )


class PasteStepperList(ModelCommand):
    """
    args:
    """
    BUFFER = 0
    ARGS_NO = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theBuffer = self.theArgs[ self.BUFFER ]
    
        if self.__theBuffer.getType() != ME_STEPPER_TYPE:
            return False
        return True

    
    def do( self ):
        # get steppers from Buffer one by one
        stepperList = self.__theBuffer.getStepperList()
        modelStepperList = self.theModel.getStepperList()
        pasteBuffer = self.theBufferFactory.createStepperListBuffer()
        deleteList = []

        for aStepper in stepperList:
            if aStepper in modelStepperList:
                # if exist ask whether to overwrite it
                msgtext = aStepper + " already exists. Overwrite?"

                if self.theReceiver.printMessage( msgtext, ME_YESNO ):

                    # if yes add to be destroyed to paste list
                    self.theBufferFactory.addToStepperListBuffer(  pasteBuffer, aStepper )

                    # delete old stepper
                    self.theModel.deleteStepper( aStepper )
                else:
                    continue

            # add to delete buffer
            deleteList.append( aStepper )

            # paste it
            self.theBufferPaster.pasteStepperListBuffer(  self.__theBuffer, aStepper )

        # create a paste and a delete reverse command
        deleteCommand = DeleteStepperList( self.theReceiver, deleteList )
        pasteCommand = PasteStepperList( self.theReceiver, pasteBuffer )
        self.theReverseCommandList = [ deleteCommand, pasteCommand ]
        return True

    def createReverseCommand( self ):
        # reverse command created through do
        self.theReverseCommandList = None

    def getAffected( self ):
        return (ME_STEPPER_TYPE, None )



class ChangeStepperClass(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    CLASSNAME = 0
    ID =1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.ID ]
        self.__theClassName = self.theArgs[ self.CLASSNAME ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        return True
    
    def do( self ):
        propertyList = self.theModel.getStepperPropertyList( self.__theID )
        coreBuffer = self.theBufferFactory.createStepperPropertyListBuffer( )
        optionalBuffer = self.theBufferFactory.createStepperPropertyListBuffer( )

        for aProperty in propertyList:
            if self.theModel.getStepperPropertyAttributes( self.__theID, aProperty)[MS_DELETEABLE_FLAG]:
                self.theBufferFactory.addToStepperPropertyListBuffer( optionalBuffer, self.__theID, aProperty)
            else:
                self.theBufferFactory.addToStepperPropertyListBuffer( coreBuffer, self.__theID, aProperty)


        aClassName = self.theModel.getStepperClassName( self.__theID )

        # delete stepper

        self.theModel.deleteStepper( self.__theID )

        # create newstepper
        self.theModel.createStepper( self.__theClassName, self.__theID )

        # paste property list
        self.theBufferPaster.pasteStepperPropertyListBuffer( coreBuffer, self.__theID, None, False )
        self.theBufferPaster.pasteStepperPropertyListBuffer( optionalBuffer, self.__theID)

        # set system properties
        systemList = coreBuffer.getProperty( ME_STEPPER_SYSTEMLIST )
        for aSystem in systemList:
            self.theModel.setEntityProperty( aSystem +':'+ ME_STEPPERID, [ self.__theID ] )
        
        # set process properties
        processList = coreBuffer.getProperty( ME_STEPPER_PROCESSLIST )
        for aProcess in processList:
            self.theModel.setEntityProperty( aProcess+':'+ ME_STEPPERID, [ self.__theID ] )

        return True
        

    def createReverseCommand( self ):
        # store stepper
        pasteBuffer = self.theBufferFactory.createStepperPropertyListBuffer(  self.__theID , self.theModel.getStepperPropertyList( self.__theID ) )

        aClassName = self.theModel.getStepperClassName( self.__theID )

        # create chg command
        changeCommand = ChangeStepperClass( self.theReceiver, aClassName, self.__theID )
    
        # create paste command
        pasteCommand = PasteStepperPropertyList( self.theReceiver, self.__theID, pasteBuffer )
        
        self.theReverseCommandList = [ changeCommand , pasteCommand ]


    def getAffected( self ):
        return (ME_STEPPER_TYPE, self.__theID )


class ChangeStepperProperty(ModelCommand):
    """
    args:
    """
    ARGS_NO = 3
    STEPPERID = 0
    PROPERTYNAME = 1
    VALUE = 2

    def checkArgs( self ):
        #check if ID, propertyname exists
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__thePropertyName = self.theArgs[ self.PROPERTYNAME]
        if self.__thePropertyName not in self.theModel.getStepperPropertyList( self.__theID ):
            return False
        # check if type is same 
        self.__theValue = self.theArgs[self.VALUE]
        if self.theModel.getStepperPropertyAttributes( self.__theID, self.__thePropertyName ) [MS_GETTABLE_FLAG]:
            self.__theoldProp = self.theModel.getStepperProperty(self.__theID, self.__thePropertyName )
        else:
            self.__theoldProp = None
        aPropertyType = self.theModel.getStepperPropertyType( self.__theID, self.__thePropertyName )
        convertedValue = DMTypeCheck( self.__theValue, aPropertyType )
        if convertedValue == None:
            return False
        else:
            self.__theValue = convertedValue
        return True


    
    def do( self ):
        self.theModel.setStepperProperty( self.__theID, self.__thePropertyName, self.__theValue )
        return True



    def createReverseCommand( self ):
        self.theReverseCommandList = None
        if self.__theoldProp != None:
            revcom = ChangeStepperProperty( self.theReceiver, self.__theID, self.__thePropertyName, self.__theoldProp )
            self.theReverseCommandList = [ revcom ]



    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )


class CreateStepperProperty(ModelCommand):
    """
    args:
    """
    ARGS_NO = 4
    STEPPERID = 0
    PROPERTYNAME = 1
    VALUE = 2
    TYPE = 3

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__thePropertyName = self.theArgs[ self.PROPERTYNAME ]
        if self.__thePropertyName in self.theModel.getStepperPropertyList( self.__theID ):
            return False
        # check if type is same 
        self.__theValue = self.theArgs[ self.VALUE ]
        self.__theType = self.theArgs[ self.TYPE ]
        convertedValue = DMTypeCheck( self.__theValue, self.__theType )
        if convertedValue != None:
            self.__theValue = convertedValue
        else:
            return False
        #CHECK WHETHER PROPERTIES CAN BE CREATED WHEN DM IS AVAILABLE!!!
        aClass = self.theReceiver.getModel().getStepperClassName( self.__theID )
        classInfoList = self.theReceiver.getDMInfo().getClassInfoList( aClass )
        if DM_ACCEPTNEWPROPERTY in classInfoList:
            return self.theReceiver.getDMInfo().getClassInfo( aClass,  DM_ACCEPTNEWPROPERTY )
        else:
            return False
        return True

    
    def do( self ):

        self.theModel.createStepperProperty( self.__theID, self.__thePropertyName, self.__theValue, self.__theType )
        return True
        


    def createReverseCommand( self ):
        revcom = DeleteStepperPropertyList( self.theReceiver, self.__theID, [ self.__thePropertyName ] )
        self.theReverseCommandList = [ revcom ]

    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )


class DeleteStepperPropertyList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    STEPPERID = 0
    PROPERTYLIST = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__thePropertyList = self.theArgs[ self.PROPERTYLIST ]
        # check args( at least one should be deleteable)
        for aProperty in self.__thePropertyList:
            if self.theModel.getStepperPropertyAttributes( self.__theID, aProperty )[ME_DELETEABLE_FLAG]:
                return True
        return False

    
    def do( self ):
        for aPropertyName in self.__thePropertyList:
            try:
                self.theModel.deleteStepperProperty( self.__theID, aPropertyName )
            except:
                pass
        return True

    def createReverseCommand( self ):
        propBuffer = self.theBufferFactory.createStepperPropertyListBuffer( self.__theID, self.__thePropertyList )
        self.theReverseCommandList = [ PasteStepperPropertyList(self.theReceiver, self.__theID, propBuffer ) ]


    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )


class RenameStepperProperty(ModelCommand):
    """
    args:
    """
    ARGS_NO = 3
    STEPPERID = 0
    OLDNAME = 1
    NEWNAME = 2

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__theOldName = self.theArgs[ self.OLDNAME ]
        if self.__theOldName not in self.theModel.getStepperPropertyList( self.__theID ):
            return False
        if not self.theModel.getStepperPropertyAttributes(self.__theID, self.__theOldName )[ME_DELETEABLE_FLAG]:
            return False
        self.__theNewName = self.theArgs[ self.NEWNAME ]
        if self.__theNewName in self.theModel.getStepperPropertyList( self.__theID ):
            return False
        #CHECK WHETHER NEW PROPERTIES CAN BE ADDED!!!
        aClass = self.theReceiver.getModel().getStepperClassName( self.__theID )
        classInfoList = self.theReceiver.getDMInfo().getClassInfoList( aClass )
        if DM_ACCEPTNEWPROPERTY in classInfoList:
            return self.theReceiver.getDMInfo().getClassInfo( aClass,  DM_ACCEPTNEWPROPERTY )
        else:
            return False
        return True

    def do( self ):
        aValue = copyValue( self.theModel.getStepperProperty( self.__theID, self.__theOldName ) )
        aType = self.theModel.getStepperPropertyType( self.__theID, self.__theOldName )
        self.theModel.deleteStepperProperty( self.__theID, self.__theOldName )
        self.theModel.createStepperProperty( self.__theID, self.__theNewName, aValue, aType )
        return True


    def createReverseCommand( self ):
        reverseCommand = RenameStepperProperty( self.theReceiver, self.__theID, self.__theNewName, self.__theOldName )
        self.theReverseCommandList = [ reverseCommand ]


    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )


class CopyStepperPropertyList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    STEPPERID = 0
    PROPERTYLIST = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__thePropertyList = self.theArgs[ self.PROPERTYLIST ]
        return True

    
    def do( self ):
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createStepperPropertyListBuffer( self.__theID, self.__thePropertyList ) )
        return True


    def createReverseCommand( self ):
        # there is no reverse command for copy
        self.theReverseCommandList = None

    def getAffected( self ):
        return ( None, None )

class CutStepperPropertyList(ModelCommand):
    """
    args:
    """

    ARGS_NO= 2
    STEPPERID = 0
    PROPERTYLIST = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__thePropertyList = self.theArgs[ self.PROPERTYLIST ]
        for aProperty in self.__thePropertyList:
            if self.theModel.getStepperPropertyAttributes( self.__theID, aProperty )[ME_DELETEABLE_FLAG]:
                return True
        return False

    
    def do( self ):
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createStepperPropertyListBuffer( self.__theID, self.__thePropertyList ) )
        for aPropertyName in self.__thePropertyList:
            try:
                self.theModel.deleteStepperProperty( self.__theID, aPropertyName)
            except Exception:
                pass
        return True


    def createReverseCommand( self ):
        propBuffer = self.theBufferFactory.createStepperPropertyListBuffer( self.__theID, self.__thePropertyList )
        self.theReverseCommandList = [ PasteStepperPropertyList(self.theReceiver, self.__theID, propBuffer ) ]


    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )


class PasteStepperPropertyList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    STEPPERID = 0
    BUFFER = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.STEPPERID ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        self.__theBuffer = self.theArgs[ self.BUFFER ]
        if self.__theBuffer.getType() != ME_PROPERTY_TYPE:
            return False            
        return True

    def do( self ):
        fullPropertyList = self.theModel.getStepperPropertyList( self.__theID )
        bufferPropertyList = self.__theBuffer.getPropertyList()
        deleteList = []
        pasteBuffer = self.theBufferFactory.createStepperPropertyListBuffer()
        #check if property exist
        for aProperty in bufferPropertyList:
            if aProperty in fullPropertyList:
                if self.theModel.getStepperPropertyAttributes(self.__theID, aProperty )[ME_SETTABLE_FLAG]:
                    # if exist try to change value
                    self.theBufferFactory.addToStepperPropertyListBuffer( pasteBuffer, self.__theID, aProperty)
                    aValue = self.__theBuffer.getProperty(aProperty)
                    aType = self.theModel.getStepperPropertyType( self.__theID, aProperty )
                    convertedValue = DMTypeCheck( aValue, aType )
                    if convertedValue != None:
                        self.theModel.setStepperProperty( self.__theID, aProperty, convertedValue )
            else:
                # if not exists paste it
                deleteList.append( aProperty)
                self.theBufferPaster.pasteStepperPropertyListBuffer( self.__theBuffer, self.__theID, aProperty )
        # create reverse command
        deleteCommand = DeleteStepperPropertyList( self.theReceiver, self.__theID, deleteList )
        pasteCommand = PasteStepperPropertyList( self.theReceiver, self.__theID, pasteBuffer )
        self.theReverseCommandList = [ deleteCommand, pasteCommand ]
        return True
        
        

    def createReverseCommand( self ):
        #reverse command is created while doing operation
        self.theReverseCommandList = None


    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )


class SetStepperInfo(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    ID = 0
    STRINGS = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.ID ]
        self.__theStrings = self.theArgs[ self.STRINGS ]
        if self.__theID not in self.theModel.getStepperList():
            return False
        return True

    
    def do( self ):
        self.theModel.setStepperInfo( self.__theID, self.__theStrings )
        return True


    def createReverseCommand( self ):
        aStrings =  self.theModel.getStepperInfo( self.__theID ) 
        self.theReverseCommandList = [ SetStepperInfo( self.theReceiver, self.__theID, aStrings) ]


    def getAffected( self ):
        return (ME_STEPPER_TYPE, self.__theID )
