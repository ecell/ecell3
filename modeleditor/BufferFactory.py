from ModelEditor import *
from Buffer import *
from DMInfo import *
from Constants import *

class BufferFactory:

    """
    this class is for creating Buffers
    """
    def __init__(self, aModel):
        self.theModel = aModel


    def sortSystemIDs( self, aFullIDList ):
        aFullIDList.sort( self.__cmpFunc )
        return aFullIDList

    def __cmpFunc( self, xID,yID):
        x = self.__getRank( xID )
        y = self.__getRank( yID )
        return (x>y)-(x<y)

            
    def __getRank( self, aFullID ):
        aPath = aFullID.split(':')[1]
        aPath = aPath.strip('/')
        return len(aPath.split('/'))


    def createSystemListBuffer( self, aFullIDList = None ):
        aBuffer = SystemListBuffer()

        if aFullIDList != None:
            aFullIDList = self.sortSystemIDs( aFullIDList )
            for anID in aFullIDList:
                self.addToSystemListBuffer( aBuffer, anID )
        return aBuffer



    def addToSystemListBuffer( self, aBuffer, aFullID ):

        #if aBuffer.isEntityExist( aFullID ):
        #   return

        # get System class from model
        aClass = self.theModel.getEntityClassName( aFullID )

        # create System in buffer
        aBuffer.createEntity( aClass, aFullID )

        # get System propertylist from model
        propertyList = self.theModel.getEntityPropertyList( aFullID )
    
        for aProperty in propertyList:

            aFullPN = createFullPN( aFullID, aProperty )

            # get System property values from model
            if self.theModel.getEntityPropertyAttributes( aFullPN)[MS_GETTABLE_FLAG]:
                aValueList = self.theModel.getEntityProperty( aFullPN )

                # create System properties in model
                aBuffer.createEntityProperty( aFullID, aProperty, aValueList )

        aType = aFullID.split(':')[0]
        if aType != ME_SYSTEM_TYPE:
            return
        aSysPath = convertSysIDToSysPath( aFullID )

        for aType in [ ME_SYSTEM_TYPE, ME_PROCESS_TYPE, ME_VARIABLE_TYPE ]:

        # get subsystems
            anIDList = self.theModel.getEntityList(  aType, aSysPath )

            for anID in anIDList:
                newFullID = aType + ':' + aSysPath + ':' + anID
                self.addToSystemListBuffer( aBuffer, newFullID )



    def incrementSystemBuffer( self, toBuffer, fromBuffer ):

        anIDList = fromBuffer.getEntityList( ME_SYSTEM_TYPE, '.' )
        for anID in anIDList:
            aFullID = ME_SYSTEM_TYPE + ':.:' + anID
            self.__incrementSystemBuffer( toBuffer, fromBuffer, aFullID )



    def __incrementSystemBuffer( self, toBuffer, fromBuffer, aFullID ):
        #if toBuffer.isEntityExist( aFullID ):
        #   return

        # get System class from model
        aClass = fromBuffer.getEntityClassName( aFullID )

        # create System in buffer
        toBuffer.createEntity( aClass, aFullID )

        # get System propertylist from model
        propertyList = fromBuffer.getEntityPropertyList( aFullID )

        for aProperty in propertyList:

            aFullPN = createFullPN( aFullID, aProperty )

            # get System property values from model

            aValueList = fromBuffer.getEntityProperty(aFullID, aProperty)
            # create System properties in model
            toBuffer.createEntityProperty( aFullID, aProperty, aValueList )

        aType = aFullID.split(':')[0]
        if aType != ME_SYSTEM_TYPE:
            return
        aSysPath = convertSysIDToSysPath( aFullID )

        for aType in [ ME_SYSTEM_TYPE, ME_PROCESS_TYPE, ME_VARIABLE_TYPE ]:

        # get subsystems
            anIDList = fromBuffer.getEntityList(  ME_SYSTEM_TYPE, aSysPath )
            for anID in anIDList:
                newFullID = aType + ':' + aSysPath + ':' + anID
                self.__incrementSystemBuffer( toBuffer, fromBuffer, aFullID )



    def createStepperListBuffer( self, anIDList = None ):
        aBuffer = StepperListBuffer()
        if anIDList != None:
            for anID in anIDList:
                self.addToStepperListBuffer( aBuffer, anID )
        return aBuffer



    def addToStepperListBuffer( self, aBuffer, anID ):
        # get stepper class from model
        aClass = self.theModel.getStepperClassName( anID )

        # create stepper in buffer
        aBuffer.createStepper( aClass, anID )

        # get stepper propertylist from model
        propertyList = self.theModel.getStepperPropertyList( anID )
    
        for aProperty in propertyList:

            # get stepper property values from model
            if self.theModel.getStepperPropertyAttributes( anID, aProperty)[MS_GETTABLE_FLAG]:
                aValueList = self.theModel.getStepperProperty( anID, aProperty )
                # create stepper properties in model
                aBuffer.createStepperProperty( anID, aProperty, aValueList )



    def incrementStepperBuffer( self, toBuffer, fromBuffer ):
        anIDList = fromBuffer.getStepperList()

        for anID in anIDList:
            #if anID in toBuffer.getStepperList():
            #   continue

            # get stepper class from frombuffer
            aClass = fromBuffer.getStepperClassName( anID )

            # create stepper in tobuffer
            toBuffer.createStepper( aClass, anID )

            # get stepper propertylist from frombuffer
            propertyList = fromBuffer.getStepperPropertyList( anID )

            for aProperty in propertyList:
        
                # get stepper property values from frombuffer
                aValueList = fromBuffer.getStepperProperty( anID, aProperty )

                # create stepper properties in tobuffer
                toBuffer.createStepperProperty( anID, aProperty, aValueList )



    def createProcessListBuffer( self, aFullIDList = None ):
        aBuffer = ProcessListBuffer()
        if aFullIDList != None:
            for aFullID in aFullIDList:
                self.addToProcessListBuffer( aBuffer, aFullID )
        return aBuffer


    def addToProcessListBuffer( self, aBuffer, aFullID ):
        # get process class from model
        anID = aFullID.split(':')[2]
        aClass = self.theModel.getEntityClassName( aFullID )

        # create process in buffer
        aBuffer.createEntity( aClass, anID )

        # get process propertylist from model
        propertyList = self.theModel.getEntityPropertyList( aFullID )
    
        for aProperty in propertyList:

            aFullPN = createFullPN( aFullID, aProperty )

            # get process property values from model
            if self.theModel.getEntityPropertyAttributes( aFullPN )[MS_GETTABLE_FLAG]:
                aValueList = self.theModel.getEntityProperty( aFullPN )

                # create stepper properties in model
                aBuffer.createEntityProperty( anID, aProperty, aValueList )



    def incrementProcessBuffer( self, toBuffer, fromBuffer ):
        anIDList = fromBuffer.getEntityList( ME_PROCESS_TYPE )

        for anID in anIDList:
            # if anID in toBuffer.getEntityList():
            # continue

            # get entity class from frombuffer
            aClass = fromBuffer.getEntityClassName( anID )

            # create entity in tobuffer
            toBuffer.createEntity( aClass, anID )

            # get entity propertylist from frombuffer
            propertyList = fromBuffer.getEntityPropertyList( anID )

            for aProperty in propertyList:

                # get entity property values from frombuffer
                aValueList = fromBuffer.getEntityPropertyList( anID)

                # create entity properties in tobuffer
                toBuffer.createEntityProperty( anID, aProperty, aValueList )



    def createVariableListBuffer( self, aFullIDList = None ):
        aBuffer = VariableListBuffer()
        if aFullIDList != None:
            for aFullID in aFullIDList:
                self.addToProcessListBuffer( aBuffer, aFullID )
        return aBuffer



    def addToVariableListBuffer( self, aBuffer, aFullID ):
        # get process class from model
        anID = aFullID.split(':')[2]
        aClass = self.theModel.getEntityClassName( aFullID )

        # create process in buffer
        aBuffer.createEntity( aClass, anID )

        # get process propertylist from model
        propertyList = self.theModel.getEntityPropertyList( aFullID )
    
        for aProperty in propertyList:

            aFullPN = createFullPN( aFullID, aProperty )

            # get process property values from model
            if self.theModel.getEntityPropertyAttributes( aFullPN )[MS_GETTABLE_FLAG]:
                aValueList = self.theModel.getEntityProperty( aFullPN )

                # create stepper properties in model
                aBuffer.createEntityProperty( anID, aProperty, aValueList )



    def incrementVariableBuffer( self, toBuffer, fromBuffer ):
        anIDList = fromBuffer.getEntityList( ME_VARIABLE_TYPE )

        for anID in anIDList:
            # if anID in toBuffer.getEntityList():
            # continue

            # get entity class from frombuffer
            aClass = fromBuffer.getEntityClassName( anID )

            # create entity in tobuffer
            toBuffer.createEntity( aClass, anID )

            # get entity propertylist from frombuffer
            propertyList = fromBuffer.getEntityPropertyList( anID )

            for aProperty in propertyList:

                # get entity property values from frombuffer
                aValueList = fromBuffer.getEntityPropertyList(anID)

                # create entity properties in tobuffer
                toBuffer.createEntityProperty( anID, aProperty, aValueList )



    def createEntityListBuffer( self, aType, aFullIDList = None ):
        if aType == ME_SYSTEM_TYPE:
            return self.createSystemListBuffer( aFullIDList )
        elif aType == ME_PROCESS_TYPE:
            return self.createProcessListBuffer ( aFullIDList )
        else:
            return self.createVariableListBuffer ( aFullIDList )



    def addToEntityListBuffer( self, aBuffer, aFullID ):
        if aBuffer.getType() == ME_SYSTEM_TYPE:
            self.addToSystemListBuffer( aBuffer, aFullID )
        elif aBuffer.getType() == ME_PROCESS_TYPE:
            self.addToProcessListBuffer( aBuffer, aFullID )
        else:
            self.addToVariableListBuffer( aBuffer, aFullID )



    def incrementEntityBuffer( self, toBuffer, fromBuffer ):
        if toBuffer.getType() == ME_SYSTEM_TYPE:
            self.incrementSystemBuffer( toBuffer, fromBuffer )
        elif toBuffer.getType() == ME_PROCESS_TYPE:
            self.incrementProcessBuffer( toBuffer, fromBuffer  )
        else:
            self.incrementVariableBuffer( toBuffer, fromBuffer  )



    def createEntityPropertyListBuffer ( self, aFullPNList = None ):
        aPropertyBuffer = PropertyListBuffer()
        if aFullPNList != None :
            for aFullPN in aFullPNList:
                self.addToEntityPropertyListBuffer( aPropertyBuffer, aFullPN )
        return aPropertyBuffer



    def addToEntityPropertyListBuffer( self, aBuffer, aFullPN ):
        if self.theModel.getEntityPropertyAttributes( aFullPN )[MS_GETTABLE_FLAG]:
            aValueList = self.theModel.getEntityProperty( aFullPN )
            aPropertyName = aFullPN.split(':')[3]
            aBuffer.createProperty( aPropertyName, aValueList )


    def incrementEntityPropertyBuffer( self, toBuffer, fromBuffer ):
        aPropertyList = fromBuffer.getPropertyList()
        for aPropertyName in aPropertyList:
            aValueList = fromBuffer.getProperty( aPropertyName )
            toBuffer.createProperty( aPropertyName, aValueList )


    def createStepperPropertyListBuffer ( self, aStepperID = None, aPropertyNameList = None ):
        aPropertyBuffer = PropertyListBuffer()
        if aStepperID != None and aPropertyNameList != None:
            for aPropertyName in aPropertyNameList:
                self.addToStepperPropertyListBuffer( aPropertyBuffer, aStepperID, aPropertyName )
        return aPropertyBuffer


    def addToStepperPropertyListBuffer( self, aBuffer, aStepperID, aPropertyName ):
        if self.theModel.getStepperPropertyAttributes( aStepperID, aPropertyName)[MS_GETTABLE_FLAG]:
            aValueList = self.theModel.getStepperProperty( aStepperID, aPropertyName )
            aBuffer.createProperty( aPropertyName, aValueList )


    def incrementStepperPropertyBuffer( self, toBuffer, fromBuffer ):
        aPropertyList = fromBuffer.getPropertyList
        for aPropertyName in aPropertyList:
            aValueList = fromBuffer.getProperty( aPropertyName )
            toBuffer.createProperty( aPropertyName, aValueList )



class BufferPaster:
    """
    this class is for pasting Buffers into the Model
    can paste the content of Buffers one by one
    """

    def __init__( self, aModel ):
        self.theModel = aModel


    def pasteSystemListBuffer ( self, aBuffer, toSystemPath, anID = None  ):

        if anID != None:
            anIDList = [ anID ]
        else:
            anIDList = aBuffer.getEntityList( ME_SYSTEM_TYPE, '.' )
        aRoot = aBuffer.getRootDir()
        aBuffer.setRootDir( toSystemPath )

        for anID in anIDList:
            toFullID = ME_SYSTEM_TYPE + ':' + toSystemPath + ':' + anID
            self.__pasteSystemListBuffer( aBuffer, toFullID )
        aBuffer.setRootDir( aRoot )


    def __pasteSystemListBuffer( self, aBuffer, aFullID ):
        # get class
        aClass = aBuffer.getEntityClassName( aFullID )
        
        # create entity
        self.theModel.createEntity( aClass, aFullID )

        # get propertylist
        propBuff = aBuffer.getPropertyBuffer( aFullID )

        # paste Entityproperty
        self.pasteEntityPropertyListBuffer( propBuff, aFullID )

        # if system
        if getFullIDType( aFullID ) != ME_SYSTEM_TYPE:
            return
        # get subsystem, property, variable list
        aSysPath = convertSysIDToSysPath( aFullID )
        for aType in [ ME_SYSTEM_TYPE, ME_PROCESS_TYPE, ME_VARIABLE_TYPE ]:
            anIDList = aBuffer.getEntityList( aType, aSysPath )
            for anID in anIDList:
                aFullID = aType + ':' + aSysPath + ':' + anID

                # call self recursively 
                self.__pasteSystemListBuffer( aBuffer, aFullID )



    def pasteEntityListBuffer ( self, aBuffer, systemPath, anID = None  ):

        if aBuffer.getType() == ME_SYSTEM_TYPE:
            self.pasteSystemListBuffer( aBuffer, systemPath, anID )
        elif aBuffer.getType() == ME_PROCESS_TYPE:
            self.pasteProcessListBuffer( aBuffer, systemPath, anID )
        else:
            self.pasteVariableListBuffer( aBuffer, systemPath, anID )
        


    def pasteStepperListBuffer ( self, aBuffer, anID = None ):
        if anID != None:
            anIDList = [ anID ]
        else:
            anIDList = aBuffer.getStepperList()

        for anID in anIDList:
            aClass = aBuffer.getStepperClassName( anID )
            self.theModel.createStepper( aClass, anID )
            propBuffer = aBuffer.getPropertyBuffer( anID )
            self.pasteStepperPropertyListBuffer( propBuffer, anID )



    def pasteProcessListBuffer ( self, aBuffer, toSystemPath, anID = None ):
        if anID != None:
            anIDList = [ anID ]
        else:
            anIDList = aBuffer.getEntityList()

        for anID in anIDList:
            aClass = aBuffer.getEntityClassName( anID )
            aFullID = ME_PROCESS_TYPE + ':' + toSystemPath + ':' + anID
            self.theModel.createEntity( aClass, aFullID )
            propBuffer = aBuffer.getPropertyBuffer( anID )
            self.pasteEntityPropertyListBuffer( propBuffer, aFullID )



    def pasteVariableListBuffer ( self, aBuffer, toSystemPath, anID = None  ):
        if anID != None:
            anIDList = [ anID ]
        else:
            anIDList = aBuffer.getEntityList()

        for anID in anIDList:
            aClass = aBuffer.getEntityClassName( anID )
            aFullID = ME_VARIABLE_TYPE + ':' + toSystemPath + ':' + anID
            self.theModel.createEntity( aClass, aFullID )
            propBuffer = aBuffer.getPropertyBuffer( anID )
            self.pasteEntityPropertyListBuffer( propBuffer, aFullID )




    def pasteStepperPropertyListBuffer ( self, aBuffer, anID, aPropertyName = None, forceCreate = True ):
        if aPropertyName != None:
            aBufferPropertyList = [ aPropertyName ]
        else:
            aBufferPropertyList = aBuffer.getPropertyList( )

        aModelPropertyList = self.theModel.getStepperPropertyList( anID)

        for aPropertyName in aBufferPropertyList:
            aValueList = aBuffer.getProperty( aPropertyName )
            if aPropertyName in aModelPropertyList:
                if self.theModel.getStepperPropertyAttributes(anID, aPropertyName)[MS_SETTABLE_FLAG]:
                    aType = self.theModel.getStepperPropertyType( anID, aPropertyName )
                    convertedValue = DMTypeCheck( aValueList, aType )
                    if convertedValue != None:
                        self.theModel.setStepperProperty( anID, aPropertyName, convertedValue )
            elif forceCreate:
                aType = DM_PROPERTY_STRING
                convertedValue = DMTypeCheck(  aValueList, aType )
                if convertedValue != None:
                    try:
                        self.theModel.createStepperProperty( anID, aPropertyName, aValueList )
                    except Exception:
                        pass



    def pasteEntityPropertyListBuffer ( self, aBuffer, aFullID, aPropertyName = None, forceCreate = True ):
        if aPropertyName != None:
            aBufferPropertyList = [ aPropertyName ]
        else:
            aBufferPropertyList = aBuffer.getPropertyList()
        aModelPropertyList = self.theModel.getEntityPropertyList( aFullID)
        for aPropertyName in aBufferPropertyList:
            aValueList = aBuffer.getProperty( aPropertyName )
            if aPropertyName in aModelPropertyList:
                aFullPN = createFullPN( aFullID, aPropertyName )
                if self.theModel.getEntityPropertyAttributes( aFullPN )[MS_SETTABLE_FLAG]:
                    aType = self.theModel.getEntityPropertyType( aFullPN )
                    convertedValue = DMTypeCheck( aValueList, aType )
                    if convertedValue != None:
                        self.theModel.setEntityProperty( aFullPN, convertedValue)
            elif forceCreate:
                aType = DM_PROPERTY_STRING
                convertedValue = DMTypeCheck(  aValueList, aType )
                if convertedValue != None:
                    self.theModel.createEntityProperty( aFullID, aPropertyName, aValueList )

