from Constants import *
from LayoutCommand import *
from EntityCommand import *
from Utils import *
from PackingStrategy import *
import string


class CommandMultiplexer:

    def __init__( self, aModelEditor, aLayoutManager ):
        self.theLayoutManager = aLayoutManager
        self.theModelEditor = aModelEditor
        

    def processCommandList( self, aCommandList ):
        returnCmdList = []

        for aCmd in aCommandList:
            if aCmd.doMultiplex:
                returnCmdList.extend(self.processCommand( aCmd ))
        return returnCmdList


    def processCommand( self, aCmd ):
        cmdType = aCmd.__class__.__name__
        returnCmdList = []
        anIterator = self.theLayoutManager.createObjectIterator()
        returnCmdList.append( aCmd )
        if cmdType == "DeleteEntityList":
            fullIDList = aCmd.theArgs[ aCmd.IDLIST ]
            returnCmdList = []
            for aFullID in fullIDList:
                returnCmdList.extend( self.__deleteObjectsByFullID( aFullID) )
            for aFullID in fullIDList:
                returnCmdList.extend( self.__deleteReferringVarrefs( aFullID ) )
            returnCmdList.append( aCmd )


        elif cmdType == "RenameEntity":
            oldFullID = aCmd.theArgs[ aCmd.OLDID ]
            newFullID = aCmd.theArgs[ aCmd.NEWID ]
            anIterator.filterByFullID( oldFullID )
        
            aType = getFullIDType( newFullID )
            while True:
                anObject = anIterator.getNextObject()
                if anObject == None:
                    break
                returnCmdList += self.__relabelObject( anObject, aType, newFullID )

                if aType == ME_SYSTEM_TYPE:
                    returnCmdList.extend( self.__changeFullID( anObject, newFullID ) )
                    
            if aType == ME_SYSTEM_TYPE:
                #check broken connections!
                returnCmdList.extend( self.__checkBrokenConnections( oldFullID, newFullID ) )               
                

        elif cmdType == "CutEntityList":
            fullIDList = aCmd.theArgs[ aCmd.IDLIST ]
            for aFullID in fullIDList:
                returnCmdList.extend( self.__deleteObjectsByFullID( aFullID) )
                
            # elif cmdType == "DeleteStepperList":
            #   stepperIDList = aCmd.theArgs[ aCmd.IDLIST ]
            #   for aStepperID in stepperIDList:
            #       anIterator.deleteFilters():
            #       anIterator.filterByProperty( OB_STEPPERID, aStepperID )
            #       while True:
            #           anObject = anIterator.getNextObject()
            #           if anObject == None:
            #               break
            #           changeCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, '' )
            #           returnCmdList.append( changeCommand )

        #elif cmdType == "RenameStepper":
        #   oldID = aCmd.theArgs[ aCmd.OLDID ]
        #   newID = aCmd.theArgs[ aCmd.NEWID ]
        #   anIterator.filterByProperty( OB_STEPPERID, oldID )
        #   while True:
        #       anObject = anIterator.getNextObject()
        #       if anObject == None:
        #           break
        #       changeCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, newID )
        #       returnCmdList.append( changeCommand )

        elif cmdType == "ChangeEntityProperty":
            chgdFullPN = aCmd.theArgs[ aCmd.FULLPN ]
            newValue = aCmd.theArgs[ aCmd.VALUE ]
            chgdFullID = getFullID( chgdFullPN )
            chgdProperty = getPropertyName( chgdFullPN )
            #if chgdProperty in [ MS_PROCESS_STEPPERID, MS_SYSTEM_STEPPERID ]:
            #   returnCmdList.extend( self.__changeStepperID(chgdFullID, newValue ) )
            if chgdProperty == MS_PROCESS_VARREFLIST:
                chgCom = returnCmdList.pop()
                returnCmdList.extend( self.__changeVariableReferenceList( chgdFullID, newValue ) )
                returnCmdList.append( chgCom )
        elif cmdType == "PasteEntityPropertyList":
            # get buffer
            chgdFullID = aCmd.theArgs[ aCmd.FULLID ]
            aBuffer = aCmd.theArgs[ aCmd.BUFFER ]
            propertyList = aBuffer.getPropertyList()
            for aProperty in propertyList:
                #if aProperty in [ MS_PROCESS_STEPPERID, MS_SYSTEM_STEPPERID ]:
                #   newValue = aBuffer.getProperty( aProperty )
                #   returnCmdList.extend( self.__changeStepperID(chgdFullID, newValue ) )
                if aProperty == MS_PROCESS_VARREFLIST:
                    newValue = aBuffer.getProperty( aProperty )
                    returnCmdList.extend( self.__changeVariableReferenceList( chgdFullID, newValue ) )
        elif cmdType == "RelocateEntity":
            targetSystemID = aCmd.theArgs[ aCmd.TARGETSYSTEM ]
            aFullID = aCmd.theArgs[ aCmd.ENTITYID ]
            anIterator.filterByFullID( aFullID )
            while True:
                anObject = anIterator.getNextObject()
                if anObject == None:
                    break
                thePackingStrategy = anObject.getLayout().thePackingStrategy
                relocateCommand = thePackingStrategy.autoMoveObject( targetSystemID, anObject.getID() )
                returnCmdList.extend( relocateCommand )
        elif cmdType == "CreateObject":
            # if type is not system or process or variable pass
            objectType = aCmd.theArgs[ aCmd.TYPE ]
            objectFullID = aCmd.theArgs[ aCmd.FULLID ]
            if objectType in [ME_VARIABLE_TYPE, ME_PROCESS_TYPE, ME_SYSTEM_TYPE ]:
                if not self.theModelEditor.getModel().isEntityExist( objectFullID ):
                    if objectType == ME_PROCESS_TYPE:
                        newClass = self.theModelEditor.getDefaultProcessClass()
                    elif objectType == ME_VARIABLE_TYPE:
                        newClass = DM_VARIABLE_CLASS
                    elif objectType == ME_SYSTEM_TYPE:
                        newClass = DM_SYSTEM_CLASS

                    createCmd = CreateEntity( self.theModelEditor, objectFullID, newClass )
                    returnCmdList.insert(0, createCmd )
        elif cmdType == "SetObjectProperty":
            chgdProperty = aCmd.theArgs[ aCmd.PROPERTYNAME ]
            chgdValue = aCmd.theArgs[ aCmd.NEWVALUE ]
            chgdID = aCmd.theArgs[ aCmd.OBJECTID ]
            theLayout = aCmd.theReceiver
            theObject = theLayout.getObject ( chgdID )
            objectType = theObject.getProperty( OB_TYPE )
            if chgdProperty == OB_SHAPE_TYPE and objectType != OB_TYPE_CONNECTION:
                newLabel = theObject.getProperty( OB_LABEL )
                if theObject.getProperty( OB_HASFULLID ):
                    aType = getFullIDType( theObject.getProperty( OB_FULLID ) )
                else:
                    aType = None
                availSpace = theObject.calcLabelParam(newLabel)

                newDimx, newDimy = self.theModelEditor.theShapePluginManager.getMinDims( theObject.getProperty( OB_TYPE), chgdValue, theObject.getGraphUtils(),newLabel )

                if availSpace<newDimx:
                    newLabel=theObject.truncateLabel(newLabel,newDimx,availSpace) 
                newDimx, newDimy = self.theModelEditor.theShapePluginManager.getMinDims( theObject.getProperty( OB_TYPE), chgdValue, theObject.getGraphUtils(),newLabel )

                oldDimx=theObject.getProperty(OB_DIMENSION_X)
                oldDimy=theObject.getProperty(OB_DIMENSION_Y)
                deltaWidth=newDimx-oldDimx
                deltaHeight=newDimy-oldDimy
                #no resize necessary for system child if newDimx is smaller
                if aType==ME_SYSTEM_TYPE:
                    noChild=len(theObject.getObjectList())
                    if noChild>0:
                        largestDimX, largestDimY=theObject.getLargestChildPosXY()
                        if largestDimX>newDimx:
                            deltaWidth=largestDimX-oldDimx
                        if largestDimY>newDimy:
                            deltaHeight=largestDimY-oldDimy
                    if deltaWidth<0:
                        deltaWidth=0
                    if deltaHeight<0:
                        deltaHeight = 0
                resizeCommand = ResizeObject(theObject.getLayout(), chgdID,0, deltaHeight, 0, deltaWidth )
                returnCmdList.append( resizeCommand )
                
            if chgdProperty in GLOBALPROPERTYSET:
                if objectType == OB_TYPE_CONNECTION:
                    aProcessID = theObject.getProperty( CO_PROCESS_ATTACHED )
                    processObject = theLayout.getObject( aProcessID )
                    aProcessFullID = processObject.getProperty( OB_FULLID )
                    aVarrefName =  theObject.getProperty( CO_NAME )
                    objectIter = self.theLayoutManager.createObjectIterator()
                    objectIter.filterByFullID( aProcessFullID )
                    while True:
                        anObject = objectIter.getNextObject()
                        if anObject == None:
                            break
                        connectionList = anObject.getProperty( PR_CONNECTIONLIST )
                        for aConID in connectionList:
                            conObject = theLayout.getObject( aConID )
                            if conObject.getProperty( CO_NAME ) == aVarrefName and aConID!= chgdID:
                                chgCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), chgdProperty, chgdValue )
                                cmdList.append( chgCommand )
                else:
                    objectFullID = theObject.getProperty( OB_FULLID )
                    objectIter = self.theLayoutManager.createObjectIterator()
                    
                    objectIter.filterByFullID( objectFullID )
                    while True:
                        
                        anObject = objectIter.getNextObject()
                        if anObject == None:
                            break
                        if anObject.getID() != chgdID:
                            
                            chgCommand = SetObjectProperty( anObject.getLayout(), anObject.getID(), chgdProperty, chgdValue )
                            cmdList.append( chgCommand )
            # here comes the shaky part
            # Commands that require Issuing modelcommands ( like renaming connections, changing coefficients)
            if chgdProperty == CO_COEF:
                processObject = theObject.getProperty( CO_PROCESS_ATTACHED )
                processFullID = processObject.getProperty( OB_FULLID )
                aVarrefName =  theObject.getProperty( CO_NAME )
                aFullPN = createFullPN( processFullID, aVarrefName )
                oldValue = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
                for aVarref in oldValue:
                    if aVarref[ MS_VARREF_NAME ] == aVarrefName:
                        aVarref[ MS_VARREF_COEF ] = chgdValue
                        break
                chgCmd = ChangeEntityProperty( aFullPN, oldValue )
                cmdList.insert(0,chgCmd)
            elif chgdProperty == CO_NAME:
                processObject = theObject.getProperty( CO_PROCESS_ATTACHED )
                processFullID = processObject.getProperty( OB_FULLID )
                aVarrefName =  theObject.getProperty( CO_NAME )
                aFullPN = createFullPN( processFullID, aVarrefName )
                oldValue = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
                for aVarref in oldValue:
                    if aVarref[ MS_VARREF_NAME ] == aVarrefName:
                        aVarref[ MS_VARREF_NAME ] = chgdValue
                        break
                chgCmd = ChangeEntityProperty( aFullPN, oldValue )
                cmdList.insert(0,chgCmd)

        elif cmdType == "MoveObject":
            # decide whether relocate or move
            aLayout = aCmd.theReceiver
            objectID = aCmd.theArgs[ aCmd.OBJECTID ]
            newX = aCmd.theArgs[ aCmd.NEWX ]
            newY = aCmd.theArgs[ aCmd.NEWY ]
            newParent = aCmd.theArgs[ aCmd.NEWPARENT ]
            if newParent != None:
                anObject = aLayout.getObject( objectID )
                oldParent = anObject.getParent()
                if oldParent != newParent:
                    # issue relocate command
                    targetSystemID = newParent.getProperty( OB_FULLID )
                    aFullID = anObject.getProperty( OB_FULLID )
                    relCmd = RelocateEntity( self.theModelEditor, aFullID, targetSystemID )
                    returnCmdList.insert( len( returnCmdList)-1, relCmd )
                    anIterator.filterByFullID( aFullID )
                    while True:
                        anObject = anIterator.getNextObject()
                        if anObject == None:
                            break
                        if anObject.getID() == objectID:
                            continue
                        thePackingStrategy = anObject.getLayout().thePackingStrategy
                        relocateCommand = thePackingStrategy.autoMoveObject( targetSystemID, anObject.getID() )
                        returnCmdList.extend( relocateCommand )
#       elif cmdType == "DeleteObject":
#           aLayout = aCmd.theReceiver
#           objectID = aCmd.theArgs[ aCmd.OBJECTID ]
#           anObject = aLayout.getObject( objectID )
#           returnCmdList.extend( self.__deleteObjectByID( anObject ) )
        elif cmdType == "PasteObject":
            # construct real fullid
            aLayout = aCmd.theReceiver
            aBuffer = aCmd.theArgs[aCmd.BUFFER]
            aParentID =aCmd.theArgs[aCmd.PARENTID]
            
            aParent = aLayout.getObject(aParentID)
            if aBuffer.__class__.__name__ == "MultiObjectBuffer":
                for aSystemBufferName in aBuffer.getSystemObjectListBuffer().getObjectBufferList():
                    aSystemBuffer = aBuffer.getSystemObjectListBuffer().getObjectBuffer( aSystemBufferName )
                    self.__pasteOneObjectBuffer( aSystemBuffer, aLayout, aParent, returnCmdList ) 
                for aBufferName in aBuffer.getObjectListBuffer().getObjectBufferList():
                    anObjectBuffer = aBuffer.getObjectListBuffer().getObjectBuffer( aBufferName )
                    self.__pasteOneObjectBuffer( anObjectBuffer, aLayout, aParent, returnCmdList ) 

            else:
                    self.__pasteOneObjectBuffer( aBuffer, aLayout, aParent, returnCmdList ) 

        elif cmdType == "CreateConnection":
            foundFlag = False
            # get processfullid
            processObjectID = aCmd.theArgs[ aCmd.PROCESSOBJECTID ]
            variableObjectID = aCmd.theArgs[ aCmd.VARIABLEOBJECTID ]
            cmdVarrefName = aCmd.theArgs[ aCmd.VARREFNAME ]
            cmdDirection = aCmd.theArgs[ aCmd.DIRECTION ]
            validFlag = True
            aProcessObject = aCmd.theReceiver.getObject( processObjectID )
            processFullID = aProcessObject.getProperty( OB_FULLID )
            aVarrefName = cmdVarrefName
            aFullPN = createFullPN ( processFullID, MS_PROCESS_VARREFLIST )
            # create command for modifying variablereferencelist if not exist
            aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
            for aVarref in aVarrefList:
                if aVarref[MS_VARREF_NAME] == aVarrefName:
                    foundFlag = True
                    break
            if not foundFlag:
                if  type( variableObjectID ) == type ( [] ):
                    variableFullID = ":.:___NOTHING"
                else:
                    processObject =  aCmd.theReceiver.getObject( processObjectID )
                    processFullID = processObject.getProperty( OB_FULLID )
                    variableObject = aCmd.theReceiver.getObject( variableObjectID )
                    variableFullID = variableObject.getProperty( OB_FULLID )

                    variableFullID = getRelativeReference (processFullID, variableFullID)


                    
                aCoef = 1
                if cmdDirection == VARIABLE_TO_PROCESS:
                    aCoef = -1
                newVarref = [ aVarrefName, variableFullID, aCoef ]
                aVarrefList.append ( newVarref )
                addCmd = ChangeEntityProperty( self.theModelEditor, aFullPN, aVarrefList )
                returnCmdList.insert( len(returnCmdList) -1, addCmd )
            
        elif cmdType == "RedirectConnection":
            # get changed and new endpoint
            
            cmdObjectID = aCmd.theArgs[ aCmd.OBJECTID ]
            cmdNewProcessObjectID = aCmd.theArgs[aCmd.NEWPROCESSOBJECTID ]
            cmdNewVariableObjectID = aCmd.theArgs[aCmd.NEWVARIABLEOBJECTID ]
            cmdNewVarrefName = aCmd.theArgs[aCmd.NEWVARREFNAME ]
            connectionObject = aCmd.theReceiver.getObject( cmdObjectID )
            aVarrefName = connectionObject.getProperty( CO_NAME )
            oldProcessID = connectionObject.getProperty( CO_PROCESS_ATTACHED )
            oldProcessObject = aCmd.theReceiver.getObject( oldProcessID )
            oldProcessFullID = oldProcessObject.getProperty( OB_FULLID )
            
            
            
            oldVariableID = connectionObject.getProperty( CO_VARIABLE_ATTACHED )
            if type( oldVariableID ) not in (type( [] ), type([] ), type(None)):
                oldVariableObject = aCmd.theReceiver.getObject( oldVariableID )
            else:
                oldVariableObject = oldVariableID           
            
            newVarrefName = cmdNewVarrefName
            if newVarrefName == None:
                newVarrefName = aVarrefName
            
            oldVarrefFullPN = createFullPN ( oldProcessFullID, MS_PROCESS_VARREFLIST )
            aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( oldVarrefFullPN ) )
            for aVarref in aVarrefList:
                if aVarref[ MS_VARREF_NAME ] == aVarrefName:
                    foundVarref = aVarref
                    break
            # get coef and isrelative flag from Varref
            #theCoef = connectionObject.getProperty( CO_COEF )
            isRelative = foundVarref[ME_VARREF_COEF]
            
            # if processID is changed, it can only be changes in the rings so no changes are necessary
            
            if cmdNewProcessObjectID != None:
                pass

            # if varrefID is changed, handle four cases
            entityCmd = None
            if cmdNewVariableObjectID != None:
                if type( cmdNewVariableObjectID) == type([]):
                    aVariableObject = cmdNewVariableObjectID
                
                doDetachment = type( oldVariableObject) != type( [] )
                doAttachment = type( cmdNewVariableObjectID) != type([])
                
                #no detachment, no attachment - do nothing
                if not doDetachment and not doAttachment:
                    entityCmd = None
                    objCommands = []

                # just detachment,  first change entity property, then detach other layouts
                elif doDetachment and not doAttachment:
                    foundVarref[ MS_VARREF_FULLID ] = ":.:___NOTHING"
                    objCommands = []
                    
                    entityCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
                    objectIter = self.theLayoutManager.createObjectIterator()
                    objectIter.filterByFullID(oldProcessFullID )
                    anOldLayoutName = aCmd.theReceiver.theName
                    while True:
                        anObject = objectIter.getNextObject()
                        
                        if anObject == None:
                            break
                        aLayout = anObject.getLayout()
                        connectionList = anObject.getProperty( PR_CONNECTIONLIST )  
                        for conObjectID in connectionList:
                            conObject = aLayout.getObject( conObjectID )
                            if conObject.getProperty( CO_NAME ) == aVarrefName:
                                if (conObject.getLayout().theName== anOldLayoutName and conObject.getID() != cmdObjectID) or conObject.getLayout().theName!= anOldLayoutName:
                                
                                    attachmentPoints = copyValue(conObject.getProperty( CO_ENDPOINT2 ) )
                                    attachmentPoints[0] += 20
                                    attachmentPoints[1] += 20
                                    redirCommand = RedirectConnection( conObject.getLayout(), conObject.getID(), None, attachmentPoints, None, None, None )
                                    objCommands.append( redirCommand )
                    

                # just attachment, first change entity varref, then attach other layouts
                elif doAttachment:
                    #new varref fullid is absolute path
                    #FIXME do Attachments with autoconnect
                    objCommands = []
                    aVariableObject = aCmd.theReceiver.getObject( cmdNewVariableObjectID )
                    newVariableFullID = aVariableObject.getProperty( OB_FULLID )
                    foundVarref[ MS_VARREF_FULLID ] = getRelativeReference (oldProcessFullID, aVariableObject.getProperty( OB_FULLID ))
                    
                    entityCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
                    
                    objectIterPro = self.theLayoutManager.createObjectIterator()
                    objectIterPro.filterByFullID(oldProcessFullID )
                    anOldLayoutName = aCmd.theReceiver.theName
                    
                    while True:
                        # quit cycle (break) if next process object is None
                        # check that this is not the original process
                        # if it is, continue
                        # search for variableobjects in the same layout where the process is

                        aProObject = objectIterPro.getNextObject()
                        if aProObject == None:
                            break
                        
                        aLayout = aProObject.getLayout()
                        aProObjectID = aProObject.getID()
                        
                        if aLayout.getName() == anOldLayoutName and aProObjectID ==oldProcessID:
                            continue
                        
                        varList = aLayout.getObjectList(OB_TYPE_VARIABLE)
                        for aVarID in varList:
                            aVarObj =  aLayout.getObject(aVarID)
                            aVarFullID = aVarObj.getProperty(OB_FULLID)
                            if aVarFullID != newVariableFullID:
                                continue
                            
                            connectionList = aProObject.getProperty( PR_CONNECTIONLIST )
                                
                            for aConID in connectionList:
                                conObject = aLayout.getObject( aConID )
                                currVarIDAttached = conObject.getProperty(CO_VARIABLE_ATTACHED)
                                if type( currVarIDAttached ) not in (type( [] ), type([] ), type(None)):
                                    currVarObj = aLayout.getObject(currVarIDAttached)
                                    currVarFullID = currVarObj.getProperty(OB_FULLID)
                                

                                else:
                                    currVarFullID = None
                                
                                if conObject.getProperty( CO_NAME ) == aVarrefName and currVarFullID != newVariableFullID:
                                    
                                    thePackingStrategy = conObject.getLayout().thePackingStrategy
                            
                                    (processRing, variableRing) = thePackingStrategy.autoConnect(aProObjectID, aVarID ,cmdNewVarrefName )
                                    redirCommand = RedirectConnection( conObject.getLayout(), conObject.getID(), None, aVarID, processRing, variableRing, None )
                                    objCommands.append( redirCommand )
                                    
                            break

                        
                        
                if entityCmd != None:
                        
                    returnCmdList.insert( len(returnCmdList)-1, entityCmd )
                returnCmdList.extend( objCommands )
                

                

            elif cmdNewVarrefName != None:
                # do rename in entity list and other places
                foundVarref[ MS_VARREF_NAME ] = cmdNewVarrefName
                objCommands = []
                objectIter = self.theLayoutManager.createObjectIterator()
                objectIter.filterByFullID(oldProcessFullID )
                while True:
                    anObject = objectIter.getNextObject()
                    if anObject == None:
                        break
                    connectionList = anObject.getProperty( PR_CONNECTIONLIST )  
                    
                    for aConID in connectionList:
                        conObject = aCmd.theReceiver.getObject( aConID )
                        if conObject.getProperty( CO_NAME ) == aVarrefName and conObject.getID() == cmdObjectID:
                            redirCommand = RedirectConnection( conObject.getLayout(), conObject.getID(), None, None, None, None, cmdNewVarrefName )
                            objCommands.append( redirCommand )
                entityCmd = ChangeEntityProperty( self.theModelEditor, oldVarrefFullPN, aVarrefList )
                returnCmdList.extend( objCommands )

        for aCommand in returnCmdList:
            aCommand.doNotMultiplex()
            aCommand.doNotMultiplexReverse()
        return returnCmdList

    def __deleteReferringVarrefs( self, aFullID ):

        cmdList = []
        if getFullIDType( aFullID ) != ME_VARIABLE_TYPE:
            return cmdList
        aModel = self.theModelEditor.theModelStore
        aProcessList = aModel.getEntityProperty( createFullPN ( aFullID , MS_VARIABLE_PROCESSLIST ) )

        for aProcess in aProcessList:
            aVarrefFullPN = createFullPN ( aProcess, MS_PROCESS_VARREFLIST )
            aVarrefList = aModel.getEntityProperty( aVarrefFullPN )
            for aVarref in aVarrefList:
                aVariableFullID = createFullIDFromVarref( aProcess, aVarref )
                if aVariableFullID == aFullID:
                    aVarrefList.remove( aVarref )
            aCommand = ChangeEntityProperty( self.theModelEditor, aVarrefFullPN, aVarrefList )
            cmdList.append( aCommand )
        return cmdList


    def __pasteOneObjectBuffer( self, aBuffer, aLayout, aParent, returnCmdList ):
        objectType = aBuffer.getProperty ( OB_TYPE )
        if objectType in [ OB_TYPE_PROCESS, OB_TYPE_VARIABLE, OB_TYPE_SYSTEM]:
            objectFullID = aBuffer.getProperty( OB_FULLID )
            parentFullID = aParent.getProperty( OB_FULLID )
            # modify buffer fullid
            # create it from parent 
            parentSysPath = convertSysIDToSysPath( parentFullID )
            fullIDList = objectFullID.split(':')
            fullIDList[1] = parentSysPath
            newFullID = ':'.join( fullIDList )
            # if fullid exists do nothing
            # else create new fullid
            if not self.theModelEditor.getModel().isEntityExist( newFullID ):
                entityBuffer = aBuffer.getEntityBuffer()
                returnCmd = PasteEntityList( self.theModelEditor, entityBuffer, parentSysPath )
                returnCmdList.insert( len( returnCmdList ) - 1, returnCmd )


    def __deleteObjectsByFullID( self, aFullID ):
        cmdList = []
        anIterator = self.theLayoutManager.createObjectIterator()
        anIterator.filterByFullID( aFullID )

        while True:
            anObject = anIterator.getNextObject()

            if anObject == None:
                break
            cmdList.extend( self.__deleteObjectByID( anObject) )
        return cmdList
    
    
    def __deleteObjectByID( self, anObject):
        # issue command for deleting object, 
        
        cmdList =[]
        deleteCommand = DeleteObject( anObject.getLayout(), anObject.getID() )
        cmdList.append( deleteCommand )
        # issue commands for deleting connections

#       objectType = anObject.getProperty( OB_TYPE )
#       if objectType == ME_VARIABLE_TYPE:
#           IDProperty = VR_CONNECTIONLIST
#       elif objectType == ME_PROCESS_TYPE:
#           IDProperty = PR_CONNECTIONLIST
#       else:
#           return cmdList
#       connectionList = anObject.getProperty( IDProperty )
#
#       for aConnection in connectionList:
#           deleteCommand = DeleteObject( aConnection.getLayout(), aConnection.getID() )
#           cmdList.append( deleteCommand )
        return cmdList



    def __changeStepperID( self, aFullID, newStepperID ):
        cmdList = []
        anIterator = self.theLayoutManager.createObjectIterator()
        anIterator.filterByFullID( aFullID )
        while True:
            anObject = anIterator.getNextObject()
            if anObject == None:
                break
            chgCmd = SetObjectProperty( anObject.getLayout(), anObject.getID(), OB_STEPPERID, newStepperID)
            cmdList.append( chgCmd )
        return cmdList
        

        
    def __changeVariableReferenceList( self, aProcessFullID, newVarrefList ):
        cmdList = []
        # get old VariablereferenceList
        aFullPN = createFullPN( aProcessFullID, MS_PROCESS_VARREFLIST )
        oldVarrefList = self.theModelEditor.getModel().getEntityProperty( aFullPN )
        # find deleted
        oldVarrefNames = []
        deleteVarrefList = []
        for anOldVarref in oldVarrefList:
            oldVarrefNames.append( anOldVarref [ ME_VARREF_NAME ] )
        for anOldVarref in oldVarrefList:
            foundInNew = False
            changedTo = None
            for aNewVarref in newVarrefList:
                if aNewVarref[ ME_VARREF_NAME ] == anOldVarref [ ME_VARREF_NAME ]:
                    foundInNew = True
                elif aNewVarref[ ME_VARREF_COEF ] == anOldVarref[ ME_VARREF_COEF ]:
                    if aNewVarref[ ME_VARREF_FULLID ] == anOldVarref[ ME_VARREF_FULLID ] and aNewVarref[ ME_VARREF_NAME ] not in oldVarrefNames:
                        changedTo = aNewVarref[ ME_VARREF_NAME ]
            if not foundInNew:
                if changedTo != None:
                    cmdList.extend( self.__changeVarrefName( aProcessFullID, anOldVarref [ MS_VARREF_NAME ], changedTo ) )
                else:
                    cmdList.extend( self.__deleteConnections( aProcessFullID, anOldVarref [ MS_VARREF_NAME ] ) )

        # cycle through all new varrefs
        for aNewVarref in newVarrefList:
            # process insertions
            if aNewVarref[ MS_VARREF_NAME ] not in oldVarrefNames:  
                # new varref inserted - find whether insertion is rename
                continue
            else:
                for anOldVarref in oldVarrefList:
                    if anOldVarref [ ME_VARREF_NAME ] == aNewVarref[ ME_VARREF_NAME ]:
                        break
            if anOldVarref[ ME_VARREF_FULLID ] != aNewVarref[ ME_VARREF_FULLID ]:
                # process redirections
                oldVariable = getAbsoluteReference( aProcessFullID, anOldVarref[ ME_VARREF_FULLID ] )
                newVariable = getAbsoluteReference( aProcessFullID, aNewVarref[ ME_VARREF_FULLID ] )
                
                cmdList.extend( self.__redirectVarref( aProcessFullID, anOldVarref [ ME_VARREF_NAME ], oldVariable, newVariable ) )
            if anOldVarref[ ME_VARREF_COEF ] != aNewVarref[ ME_VARREF_COEF ]:
                # process coef changes
                cmdList.extend( self.__changeCoef( aProcessFullID, anOldVarref [ ME_VARREF_NAME ], aNewVarref[ ME_VARREF_COEF ] ) )

        return cmdList


    def __deleteConnections( self, aProcessFullID, aVarrefName, doNotTouch = None ):
        cmdList = []
        objectIter = self.theLayoutManager.createObjectIterator()
        objectIter.filterByFullID(aProcessFullID )
        while True:
            anObject = objectIter.getNextObject()
            if anObject == None:
                break
            connectionList = anObject.getProperty( PR_CONNECTIONLIST )
            aLayout = anObject.getLayout()
            for conID in connectionList:
                conObject = aLayout.getObject( conID )
                if conObject.getProperty( CO_NAME ) == aVarrefName and conObject.getLayout().theName!= doNotTouch:
                    deleteCommand = DeleteObject( aLayout, conObject.getID() )
                    cmdList.append( deleteCommand )
        return cmdList


    def __redirectVarref( self, aProcessFullID, aVarrefName, oldVariable, newVariable ):
        cmdList = []
        objectIter = self.theLayoutManager.createObjectIterator()
        objectIter.filterByFullID( aProcessFullID )
        while True:
            anObject = objectIter.getNextObject()
            
            if anObject == None:
                break
            aProObjectID = anObject.getID()
            connectionList = anObject.getProperty( PR_CONNECTIONLIST )
            aLayout = anObject.getLayout()  
            for conID in connectionList:
                conObject = aLayout.getObject( conID )
                if conObject.getProperty( CO_NAME ) == aVarrefName:
                    if newVariable ==None:
                        newVariableID = conObject.getProperty( CO_ENDPOINT2)
                        newVariableID[0]+=20
                        newVariableID[1]+=20
                        proRing = None
                        varRing = None
                    else:
                        newVariableID = conObject.getProperty( CO_VARIABLE_ATTACHED)
                        proRing = conObject.getProperty( CO_PROCESS_RING)
                        varRing = conObject.getProperty( CO_VARIABLE_RING)
                    
                    redirCommand = RedirectConnection( conObject.getLayout(), conObject.getID(), aProObjectID, newVariableID, proRing, varRing, aVarrefName )
                    cmdList.append( redirCommand )
                        
        return cmdList


    def __changeCoef( self, aProcessFullID, aVarrefName, newCoef ):
        cmdList = []
        objectIter = self.theLayoutManager.createObjectIterator()
        objectIter.filterByFullID( aProcessFullID )
        while True:
            anObject = objectIter.getNextObject()
            if anObject == None:
                break
            aLayout = anObject.getLayout()
            connectionList = anObject.getProperty( PR_CONNECTIONLIST )  
            for conID in connectionList:
                conObject = aLayout.getObject( conID )
                if conObject.getProperty( CO_NAME ) == aVarrefName:
                    deleteCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), CO_COEF, newCoef )
                    cmdList.append( deleteCommand )

        return cmdList

    def __changeVarrefName(self, aProcessFullID, oldVarrefName, newVarrefName ):
        cmdList = []
        objectIter = self.theLayoutManager.createObjectIterator()
        objectIter.filterByFullID( aProcessFullID )
        while True:
            anObject = objectIter.getNextObject()
            if anObject == None:
                break
            aLayout = anObject.getLayout()
            connectionList = anObject.getProperty( PR_CONNECTIONLIST )  
            for conID in connectionList:
                conObject = aLayout.getObject( conID )
                if conObject.getProperty( CO_NAME ) == oldVarrefName:
                    renameCommand = SetObjectProperty( conObject.getLayout(), conObject.getID(), CO_NAME, newVarrefName )
                    cmdList.append( renameCommand )

        return cmdList
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\\
    def __changeFullID( self, aSystemObject, newFullID ):
        returnCmdList = []
        newPath = convertSysIDToSysPath( newFullID )
        aLayout = aSystemObject.getLayout()
        for anID in aSystemObject.getObjectList():
            anObject = aLayout.getObject( anID )
            aType = getFullIDType( newFullID )
            if not anObject.getProperty( OB_HASFULLID ):
                continue
            aFullID = anObject.getProperty( OB_FULLID )
            ( aType, aPath, aName ) = aFullID.split(':')
            aFullID = ':'.join( [aType, newPath, aName ] )
            returnCmdList += self.__relabelObject ( anObject, aType, aFullID )

            if aType == ME_SYSTEM_TYPE:

                returnCmdList.extend( self.__changeFullID( anObject, aFullID  ) )
        return returnCmdList

    def __relabelObject(  self, anObject, aType, aFullID ):
            #Label
            returnCmdList = []
            newLabel=aFullID
            if aType != ME_SYSTEM_TYPE:
                newLabel=newLabel.split(':')[2]
            anObject.calcLabelParam(newLabel)
            totalWidth,limit=anObject.getLabelParam()
            if totalWidth>limit:
                newLabel=anObject.truncateLabel(newLabel,totalWidth,limit)      
            newDimx=anObject.estLabelWidth(newLabel)
            oldDimx=anObject.getProperty(OB_DIMENSION_X)
            deltaWidth=newDimx-oldDimx
            #no resize necessary for system child if newDimx is smaller
            if aType==ME_SYSTEM_TYPE:
                noChild=len(anObject.getObjectList())
                if noChild>0:
                    largestDim=anObject.getLargestChildPosX()
                    if largestDim>newDimx:
                        deltaWidth=largestDim-oldDimx
                if deltaWidth<0:
                    deltaWidth=0
            aLayout = anObject.getLayout()
            anID = anObject.getID()
            resizeCommand = ResizeObject(aLayout, anID,0, 0, 0, deltaWidth )
            renameCommand = SetObjectProperty( aLayout, anID, OB_FULLID, aFullID )
            relabelCommand = SetObjectProperty( aLayout, anID,OB_LABEL, newLabel)
            returnCmdList.append( resizeCommand )
            returnCmdList.append( renameCommand )
            returnCmdList.append( relabelCommand )
            return returnCmdList




    def __checkBrokenConnections(self,oldFullID, newFullID ):
        returnCmdList =[]
        
        oldName = oldFullID.split(':')[2]
        # should find whether there are broken connections when renaming system and copying, cutting entities.
        sysPath = convertSysIDToSysPath( oldFullID)
        variableNameList = self.theModelEditor.getModel().getEntityList( ME_VARIABLE_TYPE, sysPath )
        variableList = createFullIDList( ME_VARIABLE_TYPE, oldFullID, variableNameList )
        
        for aVariableFullID in variableList:
            # create the variablefullId after system renamed
            aFullPN = createFullPN ( aVariableFullID, MS_VARIABLE_PROCESSLIST)
            aProcessList = copyValue( self.theModelEditor.getModel().getEntityProperty( aFullPN ) )
            (t, variablePath, variableName) = aVariableFullID.split(':')
            for aProcessFullID in aProcessList:
                aProFullPN = createFullPN ( aProcessFullID, MS_PROCESS_VARREFLIST )
                aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aProFullPN) )
                for aVarref in aVarrefList:
                    aFullID = aVarref[ME_VARREF_FULLID ]
                    refValid = True
                    if not isAbsoluteReference( aFullID ):
                        try:
                            absFullID = getAbsoluteReference( aProcessFullID, aFullID )
                        except:
                            refValid = False
                    else:
                        absFullID = aFullID
                    if absFullID.split(':')[1] == variablePath and absFullID.split(':')[2] == variableName:
                        varrefPath = aFullID.split(':')[1]
                        if varrefPath.endswith( oldName ):
                            returnCmdList.extend( self.__redirectVarref(  aProcessFullID,aVarref [ ME_VARREF_NAME ],aVarref[ME_VARREF_FULLID], None  ))
                            
        #check for child systems
        systemNameList = self.theModelEditor.getModel().getEntityList( ME_SYSTEM_TYPE, sysPath )
        systemList = createFullIDList( ME_SYSTEM_TYPE, oldFullID, systemNameList )  
        if systemList != []:
            for anOldSystemFullID in systemList:
                aNewFullIDList = newFullID.split(':')
                if aNewFullIDList[1] == '/':
                    anewName = ''.join(aNewFullIDList[1:])
                else:
                    anewName = '/'.join(aNewFullIDList[1:])
                aNewSystemFullID = anOldSystemFullID.split(':')
                aNewSystemFullID[1]=anewName    
                aNewSystemFullID = string.join(aNewSystemFullID,':')    
                returnCmdList.extend( self.__checkBrokenConnections(anOldSystemFullID, aNewSystemFullID ) )
                    
        return returnCmdList
        

        
