
from ecell.gui_config import *  # added by hiep
import ConfigParser


import tempfile

from Constants import *
import os
import os.path
import gtk
from ConfirmWindow import *  # added by hiep
from Command import *
from CommandQueue import *
from EntityCommand import *
from StepperCommand import *
import string
import traceback
import sys
from ModelStore import *
import time
import AutosaveWindow

from LayoutManagerWindow import *
from MainWindow import *
from StepperWindow import *
from EntityListWindow import *
from DMInfo import *
from PopupMenu import *
from PathwayEditor import *
from LayoutManager import *
from AboutModelEditor import *

import AutosaveWindow # added by hiep

from CommandMultiplexer import *
from ObjectEditorWindow import *
from ConnectionObjectEditorWindow import *
from LayoutEml import *
from GraphicalUtils import *

from Error import *
import time
RECENTFILELIST_FILENAME = '.modeleditor' + os.sep + '.recentlist'
RECENTFILELIST_DIRNAME = '.modeleditor'
RECENTFILELIST_MAXFILES = 10


class ModelEditor:

    """
    class for holding together Model, Command, Buffer classes
    loads and saves model
    handles undo queue
    display messages
    """

    def __init__( self, aFileName = None ):
  
        self.theConfigDB=ConfigParser.ConfigParser()
        self.theIniFileName = GUI_HOMEDIR + os.sep + '.modeleditor' + os.sep + 'preferences.ini'

        GUI_ME_PATH = os.environ['MEPATH']

        theDefaultIniFileName = GUI_ME_PATH + os.sep + 'preferences.ini'
        
        if not os.path.isfile( self.theIniFileName ):
            # get from default
            self.theConfigDB.read(theDefaultIniFileName)
            # try to write into home dir
            self.saveParameters()
        else:
            self.theConfigDB.read(self.theIniFileName)


        self.copyBuffer = None

        # set up load and save dir
        self.loadDirName = os.getcwd()
        self.saveDirName = os.getcwd()

        self.theDMInfo = DMInfo()

        # create variables
        self.__loadRecentFileList()

        self.theUpdateInterval = 0
        self.operationCount = 0
        self.theTimer = None

        self.__theModel = None
        self.theModelName = ''
        self.theModelFileName = ''
        self.theModelTmpFileName = ''
        self.operationCount=1
        

        self.theStepperWindowList = []
        self.theEntityListWindowList = []
        self.thePathwayEditorList = []
        self.theLayoutManagerWindow=None 
        self.theObjectEditorWindow = None
        self.theConnObjectEditorWindow = None
        self.theAboutModelEditorWindow = None
        self.theFullIDBrowser = None
        self.thePopupMenu = PopupMenu( self )
        self.theMainWindow = MainWindow( self )
        self.changesSaved = True
        self.openAboutModelEditor = False

        #self.openAutosaveWindow = False
        #self.theAutosaveWindow = None

        # create untitled model
        self.createNewModel()

        # set up windows
        self.theMainWindow.openWindow()
        self.theGraphicalUtils = GraphicalUtils( self.theMainWindow )
        
        # load file
        if aFileName != None:
            self.loadModel( aFileName )
        else:
            layoutName = self.theLayoutManager.getUniqueLayoutName()
            aCommand = CreateLayout( self.theLayoutManager, layoutName, True )
            self.doCommandList( [ aCommand ] )


    def quitApplication( self ):
        if not self.__closeModel():
            return False
        gtk.mainquit()
        return True

    
    def createNewModel( self ):
        """ 
        in:
        returns True if successful, false if not
        """
        
       
        # create new Model
        self.__createModel()
        

        
        self.theModelStore.createStepper(DE_DEFAULT_STEPPER_CLASS, DE_DEFAULT_STEPPER_NAME)
        self.theModelStore.setEntityProperty( 'System::/:StepperID', DE_DEFAULT_STEPPER_NAME)
        newFullID = ':'.join( [ ME_VARIABLE_TYPE, '/', 'SIZE' ] )
        self.theModelStore.createEntity( ME_VARIABLE_TYPE, newFullID )
    


        self.updateWindows()

        # create default Pathway Editor
        if self.theMainWindow.exists():
            layoutName = self.theLayoutManager.getUniqueLayoutName()
            aCommand = CreateLayout( self.theLayoutManager, layoutName, True )
            self.doCommandList( [ aCommand ] )


    def validateModel( self ):
        self.printMessage("Sorry, not implemented!", ME_ERROR )



           

    def convertSbmlToEml(self, aFileName):
        tmpFileBaseName = os.path.basename(aFileName)   
        tmpFileName = os.path.splitext( tmpFileBaseName )[0]
        tmpExt = os.path.splitext( tmpFileBaseName )[1]
        
        #Output filename 
        tmpaFileName = str(self.__getTmpDir()) + os.sep + tmpFileName + '.eml' 
        
        if string.lower(tmpExt) == '.sbml':
            os.spawnlp( os.P_WAIT, 'ecell3-sbml2eml', 'ecell3-sbml2eml', '-o', tmpaFileName, aFileName )
            return tmpaFileName        

    def convertEmlToSbml(self, aFileName):
        tmpFileBaseName = os.path.basename(aFileName)
        tmpExt = os.path.splitext( tmpFileBaseName )[1]
        tmpaFileName = self.theModelTmpFileName
       
        if tmpExt == '.eml':
            aFileName = aFileName[:-3] + string.replace(aFileName[-3:], 'eml','sbml')
            os.spawnlp( os.P_WAIT, 'ecell3-eml2sbml', 'ecell3-eml2sbml', '-o', aFileName, tmpaFileName )
        
    

    def convertEmToEml(self, aFileName):
        
        tmpFileBaseName = os.path.basename(aFileName)
        tmpExt = os.path.splitext( tmpFileBaseName )[1]
        tmpaFileName = str(self.__getTmpDir()) + os.sep + tmpFileBaseName + 'l'
        
        if tmpExt == '.em':
            os.spawnlp( os.P_WAIT, 'ecell3-em2eml', 'ecell3-em2eml', '-o', tmpaFileName, aFileName )
            return tmpaFileName

    
    def convertEmlToEm(self, aFileName):
           
        tmpFileBaseName = os.path.basename(aFileName)
        tmpExt = os.path.splitext( tmpFileBaseName )[1]
        tmpaFileName = self.theModelTmpFileName
       
        if tmpExt == '.eml':
            aFileName = aFileName[:-3] + string.replace(aFileName[-3:], 'eml','em')

            if os.path.exists(aFileName):
                if self.printMessage("%s already exists. Do you want to replace it?"%aFileName ,ME_OKCANCEL) == ME_RESULT_OK:
                    os.spawnlp( os.P_WAIT, 'ecell3-eml2em', 'ecell3-eml2em','-f', '-o', aFileName, tmpaFileName )
                else:
                    return False
            else:
                os.spawnlp( os.P_WAIT, 'ecell3-eml2em', 'ecell3-eml2em', '-o', aFileName, tmpaFileName )





    def loadEmlAndLeml(self, aFileName):
        # create new model
        self.__createModel()

        # tries to parse file
        try:
            aFileObject = open( aFileName, 'r' )
            anEml = Eml( aFileObject )
            # calls load methods
            self.__loadStepper( anEml )
            self.__loadEntity( anEml )
            self.__loadProperty( anEml )
        except:
            #display message dialog
            
            self.printMessage( "Error loading file %s"%aFileName, ME_ERROR )
            
            if str(sys.exc_type)=="Error.ClassNotExistError":
                anErrorMessage= string.join( sys.exc_value )
            else:
                anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
                    sys.exc_traceback), '\n' )
            
            self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
            return False
        # load layouts
        try:
            self.loadLayoutEml( os.path.split( aFileName ) )
        except:
            #display message dialog
            self.printMessage( "Error loading layout information from file %s"%aFileName, ME_ERROR )
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
                    sys.exc_traceback), '\n' )
            self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
        return True


    def saveEmlAndLeml(self, aFileName):

        #aFileName = self.filenameFormatter(aFileName)
        anEml = Eml()

        self.__saveStepper( anEml )

        # creates root entity
        anEml.createEntity('System', 'System::/')
        # calls save methods
        self.__saveEntity( anEml )
        self.__saveProperty( anEml )
        # add comment
        aCurrentInfo = '''<!-- created by ecell ModelEditor
 date: %s

-->
<eml>
''' % time.asctime( time.localtime() )

        aString = anEml.asString()
        anEml.destroy()
        aBuffer = aString #+ aCurrentInfo

        try:
            aFileObject = open( aFileName, 'w' )
            aFileObject.write( aBuffer )
            aFileObject.close()
        except:
            #display message dialog
            self.printMessage( "Error saving file %s"%aFileName, ME_ERROR )
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
                    sys.exc_traceback), '\n' )
            self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
            self.theMainWindow.resetCursor()
            return False
        # save layout eml
        self.saveLayoutEml(os.path.split( aFileName ))


    def loadModel ( self, aFileName = None ):
        """
        in: nothing
        returns nothing
        """
         
        
        # check if it is dir or/and file
        if os.path.isdir( aFileName ):
            self.loadDirName = aFileName.rstrip('/')
            return
        if not os.path.isfile( aFileName ):
            self.printMessage("%s file cannot be found!"%aFileName, ME_ERROR )
            return

        self.theMainWindow.displayHourglass()

        self.loadDirName = os.path.split( aFileName )[0]

        if self.loadDirName != '':
            os.chdir(self.loadDirName )
        
        aFileBaseName = os.path.basename(aFileName)
        anExt = os.path.splitext(aFileBaseName)[1]

        if anExt == '.em':
            aTmpFileName = self.convertEmToEml(aFileName)
            # aTmpFileName has .eml extension
            if not self.loadEmlAndLeml(aTmpFileName):
                self.theMainWindow.resetCursor()
                self.__createModel()
                return
                
            self.theModelTmpFileName = aTmpFileName
            # Remove the model residing in /tmp
            os.remove(self.theModelTmpFileName)

        elif anExt == '.sbml':
            pass
            """
            aTmpFileName = self.convertSbmlToEml(aFileName)
            self.loadEmlAndLeml(aTmpFileName)
            self.theModelTmpFileName = aTmpFileName
            os.remove(self.theModelTmpFileName)
            """ 
        #try to load as .eml
        else: #anExt == '.eml':
            if not self.loadEmlAndLeml(aFileName):
                self.theMainWindow.resetCursor()
                self.__createModel()
                return
            
        # log to recent list
        self.__addToRecentFileList( aFileName )
        
        if anExt == '.em':
            self.theModelFileName = aFileName + 'l'
        else:
            self.theModelFileName = aFileName

        self.theModelName = os.path.split( aFileName )[1]
        self.modelHasName = True
        self.changesSaved = True
        self.printMessage("Model %s loaded successfully."%aFileName )
        self.updateWindows()
       
        #self.theMainWindow['combo1'].set_popdown_strings(self.theLayoutManager.getLayoutNameList())

        self.theMainWindow.resetCursor()
        aLayoutList = self.theLayoutManager.getLayoutNameList()
        if len( aLayoutList ) == 0:
            layoutName = self.theLayoutManager.getUniqueLayoutName()
            aCommand = CreateLayout( self.theLayoutManager, layoutName, True )
            self.doCommandList( [ aCommand ] )
        else:
            layoutName = aLayoutList[0]
            aLayout = self.theLayoutManager.getLayout( layoutName ) 
            self.createPathwayEditor( aLayout )




    def saveModel( self,  aFileName = None ):
        """
        in: bool saveAsFlag
        returns nothing
        """
        

        # check if it is dir
        if os.path.isdir( aFileName ):
            self.saveDirName = aFileName.rstrip('/')
            return

        self.theMainWindow.displayHourglass()

        self.saveDirName = os.path.split( aFileName )[0]

        aFileBaseName = os.path.basename(aFileName)
        anExt = os.path.splitext(aFileBaseName)[1]

        if anExt == '.em':  
            self.__setEmlModelTmpFileName(1,aFileName)

            self.saveEmlAndLeml(self.theModelTmpFileName)

            self.saveLayoutEml(os.path.split( aFileName ))

            if self.convertEmlToEm(aFileName + 'l') == False:
                os.remove(self.theModelTmpFileName)
                os.remove(os.path.splitext(self.theModelTmpFileName)[0] + '.leml')
                self.theMainWindow.resetCursor()
                return
        
            self.printMessage("Some data may be lost in the 'leml' file in a '.em' saving process. (Changes Saved)", ME_WARNING)
            os.remove(self.theModelTmpFileName)
            os.remove(os.path.splitext(self.theModelTmpFileName)[0] + '.leml')

        elif anExt == '.sbml':
            # Call self.__setEmlModelTmpFileName to set abs tmp path
            # Call saveEmlAndLeml to save in abs tmp path (update)
            #self.convertEmlToSbml()
            # Prompt if there is already existing sbml file in dir 
            # Use os.remove to delete the files use to update in /tmp
            pass

        else:

            self.saveEmlAndLeml(aFileName) 

        # log to recent list
        self.__addToRecentFileList ( aFileName )
        self.theModelFileName = aFileName
        self.printMessage("Model %s saved successfully."%aFileName )
        self.theModelName = os.path.split( aFileName )[1]
        self.modelHasName = True
        self.changesSaved = True
        self.updateWindows()
        
        self.theMainWindow.resetCursor()


    def filenameFormatter(self, aFileName):
        anExt = string.lower(os.path.splitext(aFileName)[1])
        if anExt == '.sbml':
            return aFileName

        if anExt == '.leml':
            aFileName = aFileName[:-5] + string.replace(os.path.splitext(aFileName)[1], os.path.splitext(aFileName)[1],'.eml')
            return aFileName
    
        elif anExt == '.eml' or anExt =='.em':
            return aFileName

        else:
            aFileName = aFileName + '.eml'
            return aFileName        
                    
       
    def lemlExist(self,fileName):
        if os.path.isfile(fileName):
            return True
        else:
            return False

    def loadLayoutEml( self, (emlPath, emlName) ):
        
        fileName = os.path.join( emlPath, self.__getLemlFromEml( emlName ) )
        if not self.lemlExist(fileName):
            return
        fileObject = open(fileName, "r")
        aLayoutEml = LayoutEml(fileObject)

        aLayoutNameList =  aLayoutEml.getLayoutList()
        # create layouts

        for aLayoutName in  aLayoutNameList:
            self.theLayoutManager.createLayout(aLayoutName)
            aLayout = self.theLayoutManager.getLayout(aLayoutName)
        # create layoutproperties
        
            propList = aLayoutEml.getLayoutPropertyList(aLayoutName)
            for aProp in propList:
                aPropValue =aLayoutEml.getLayoutProperty(aLayoutName,aProp)
                if aPropValue == "False":
                    aPropValue=False
                elif aPropValue == "True":
                    aPropValue=True
                aLayout.setProperty(aProp,aPropValue)
                
            aRootID = aLayout.getProperty( LO_ROOT_SYSTEM )
            self.__loadObject( aLayout, aRootID, aLayoutEml, None)
            aConnObjectList = aLayoutEml.getObjectList(OB_TYPE_CONNECTION, aLayoutName)
            # finally read and create connections for layout
            for aConnID in aConnObjectList:
                aProAttachedID = aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_PROCESS_ATTACHED)
                aVarAttachedID =aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_VARIABLE_ATTACHED)
                aProRing = aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_PROCESS_RING)
                aVarRing = aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_VARIABLE_RING)
                aVarrefName =  aLayoutEml.getObjectProperty(aLayoutName, aConnID,CO_NAME)
                aLayout.createConnectionObject(aConnID, aProAttachedID, aVarAttachedID,  aProRing, aVarRing,PROCESS_TO_VARIABLE, aVarrefName )
                anObject =aLayout.getObject(aConnID)
                propList = aLayoutEml.getObjectPropertyList(aLayoutName, aConnID )
                for aProp in propList:
                    if aProp in (CO_PROCESS_ATTACHED, CO_VARIABLE_ATTACHED,CO_PROCESS_RING,CO_VARIABLE_RING, CO_NAME):
                        continue
                    aPropValue = aLayoutEml.getObjectProperty(aLayoutName, aConnID,aProp)
                    if aPropValue == None:
                        continue
                    if aPropValue=="False":
                        aPropValue=False
                    elif aPropValue=="True":
                        aPropValue=True
                    anObject.setProperty(aProp,aPropValue)



    def __loadObject( self, aLayout, anObjectID,  aLayoutEml, parentSys ):
        aLayoutName = aLayout.getName()
        propList = aLayoutEml.getObjectPropertyList(aLayoutName, anObjectID )

        objectType = aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_TYPE)
        aFullID = aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_FULLID)
        if objectType != OB_TYPE_CONNECTION:
            x=aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_POS_X)
            y=aLayoutEml.getObjectProperty(aLayoutName, anObjectID ,OB_POS_Y)

            aLayout.createObject(anObjectID, objectType, aFullID,x, y, parentSys )
            anObject =aLayout.getObject(anObjectID)
            for aProp in propList:
                if aProp in ( OB_TYPE,OB_FULLID,OB_POS_X,OB_POS_Y) :
                    continue
                aPropValue = aLayoutEml.getObjectProperty(aLayoutName, anObjectID,aProp)
                if aPropValue == None :
                    continue
                if aPropValue == "False":
                    aPropValue = False
                elif aPropValue == "True":
                    aPropValue = True
                anObject.setProperty(aProp,aPropValue)
        # create subobjects
        if objectType == OB_TYPE_SYSTEM:
            anObjectList = self.__getObjectList(aLayoutEml,aLayoutName, anObjectID)
            for anID in anObjectList:
                self.__loadObject( aLayout, anID, aLayoutEml,anObject )
        
        
    def __getObjectList(self,aLayoutEml,aLayoutName,aParentID):
        anObjectList=[]
        aProObjectList = aLayoutEml.getObjectList(OB_TYPE_PROCESS, aLayoutName, aParentID)
        aVarObjectList = aLayoutEml.getObjectList(OB_TYPE_VARIABLE, aLayoutName, aParentID)
        aSysObjectList = aLayoutEml.getObjectList(OB_TYPE_SYSTEM, aLayoutName, aParentID)
        aTextObjectList = aLayoutEml.getObjectList(OB_TYPE_TEXT, aLayoutName, aParentID)
        anObjectList = aProObjectList + aVarObjectList + aSysObjectList+ aTextObjectList
        return anObjectList



    def saveLayoutEml( self, (emlPath, emlName) ):
        fileName = os.path.join( emlPath, self.__getLemlFromEml( emlName ) )
        aLayoutEml = LayoutEml()
        # create layouts
        for aLayoutName in self.theLayoutManager.getLayoutNameList():
            self.__saveLayout( aLayoutName, aLayoutEml )

        #aCurrentInfo = '''<!-- created by ecell ModelEditor
# date: %s
#
#-->
#<leml>
#''' % time.asctime( time.localtime() )
        aString = aLayoutEml.asString()
        aLayoutEml.destroy()
#       aBuffer = string.join( string.split(aString, '<leml>\n'), aCurrentInfo)
        aBuffer = aString
        try:
            aFileObject = open( fileName, 'w' )
            aFileObject.write( aBuffer )
            aFileObject.close()
        except:
            #display message dialog
            self.printMessage( "Error saving file %s"%fileName, ME_ERROR )
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
                    sys.exc_traceback), '\n' )
            self.printMessage( anErrorMessage, ME_PLAINMESSAGE )
            



    def __saveLayout( self, aLayoutName, aLayoutEml ):
        aLayoutEml.createLayout( aLayoutName )
        # save properties
        aLayoutObject = self.theLayoutManager.getLayout( aLayoutName )
        propList = aLayoutObject.getPropertyList()
        for aProperty in propList:
            aValue = aLayoutObject.getProperty( aProperty)
            aLayoutEml.setLayoutProperty( aLayoutName, aProperty, aValue )
        # save objects
        aRootID = aLayoutObject.getProperty( LO_ROOT_SYSTEM )
        self.__saveObject( aLayoutObject, aRootID, aLayoutEml )
        # save connections
        conList = aLayoutObject.getObjectList( OB_TYPE_CONNECTION )
        for aConID in conList:
            self.__saveObject( aLayoutObject, aConID, aLayoutEml )


    def __saveObject( self, aLayoutObject, anObjectID,  aLayoutEml, parentID = 'layout' ):
        aLayoutName = aLayoutObject.getName()
        # create object
        anObject = aLayoutObject.getObject( anObjectID )
        aType = anObject.getProperty( OB_TYPE )
        aLayoutEml.createObject(  aType, aLayoutName, anObjectID, parentID )

        # save properties
        propList = anObject.getPropertyList()
        for aProperty in propList:
            if aProperty in ( PR_CONNECTIONLIST, VR_CONNECTIONLIST ):
                continue
            aValue = anObject.getProperty( aProperty)

            aLayoutEml.setObjectProperty( aLayoutName, anObjectID, aProperty, aValue )

        # save subobjects
        if aType == OB_TYPE_SYSTEM:
            for anID in anObject.getObjectList():
                self.__saveObject( aLayoutObject, anID, aLayoutEml, anObjectID )
            
        


    def __getLemlFromEml( self, emlName ):
        trunk = emlName.split('.')
        if trunk[ len(trunk)-1] == 'eml':
            trunk.pop()
        return '.'.join( trunk ) + '.leml'
    

    def closeModel( self ):
        """
        in: nothing
        return: nothing
        """
        # call close model
        self.__closeModel()



    def getUniqueEntityName( self, aType, aSystemPath ):
        """
        in: string aType, string aSystemPath
        returns string entityname or none if atype or asystempath doesnot exists
        """

        # get entitylist in system
        anEntityNameList  = self.theModelStore.getEntityList( aType, aSystemPath )

        # call getUniqueName
        return self.__getUniqueName( aType, anEntityNameList )
        



    def getUniqueStepperName(self ):
        """
        in: nothing
        return string stepper name 
        """
        # get stepper list
        aStepperIDList = self.theModelStore.getStepperList()

        # call getUniquename
        return self.__getUniqueName( 'Stepper', aStepperIDList )




    def getUniqueEntityPropertyName( self, aFullID ):
        """
        in: string aFullID
        returns new unique propertyname
        """
        # get Propertylist
        aPropertyList = self.theModelStore.getEntityPropertyList( aFullID )

        # call getUniquename
        return self.__getUniqueName( 'Property', aPropertyList )


    def getUniqueStepperPropertyName( self, anID ):
        """
        in: string anIDon_LayoutButton_toggled
        returns new unique propertyname
        """
        # get Propertylist
        aPropertyList = self.theModelStore.getStepperPropertyList( anID )

        # call getUniquename
        return self.__getUniqueName( 'Property', aPropertyList )


    def getDMInfo( self ):
        return self.theDMInfo


    def getDefaultStepperClass( self ):
        return self.theDMInfo.getClassList( ME_STEPPER_TYPE )[0]


    def getDefaultProcessClass( self ):
        return self.theDMInfo.getClassList( ME_PROCESS_TYPE )[0]


    def getRecentFileList(self):
        """
        in: nothing
        returns list of recently saved/loaded files
        """
        return self.__theRecentFileList



    def getModel( self ):
        return self.theModelStore



    def printMessage( self, aMessageText, aType = ME_PLAINMESSAGE ):
        """
        Types:
        confirmation
        plain message
        alert
        """
        if aType == ME_PLAINMESSAGE:
            if self.theMainWindow.exists():
                self.theMainWindow.displayMessage( aMessageText )
            else:
                print aMessage 
        elif aType == ME_OKCANCEL:
            msgWin = ConfirmWindow( OKCANCEL_MODE, aMessageText, 'Message')
            return msgWin.return_result()
        elif aType == ME_YESNO:
            msgWin = ConfirmWindow( OKCANCEL_MODE, aMessageText, 'Message')
            return msgWin.return_result()
        elif aType == ME_WARNING:
            msgWin = ConfirmWindow( OK_MODE, aMessageText, 'Warning!')
        elif aType == ME_ERROR:
            msgWin = ConfirmWindow( OK_MODE, aMessageText, 'ERROR!')

    def createPopupMenu( self, aComponent, anEvent ):
        self.setLastUsedComponent( aComponent )

        self.thePopupMenu.open(  anEvent )



    def doCommandList( self, aCommandList ):
#        t=[["start", time.time()]]
        
        undoCommandList = []
        aCommandList = self.theMultiplexer.processCommandList( aCommandList )
        for aCommand in aCommandList:
            # execute commands
            aCommand.execute()
#            t+=[["execute "+aCommand.__class__.__name__, time.time()]]
            ( aType, anID ) = aCommand.getAffectedObject()
            
            self.updateWindows( aType, anID )
            ( aType, anID ) = aCommand.getSecondAffectedObject()
            if aType == None and anID == None:
                pass
            else:
                self.updateWindows( aType, anID )
#                t+=[["updatewindows" + Type + " , " + anID, time.time()]]
            # get undocommands
            undoCommand = aCommand.getReverseCommandList()
            
            # put undocommands into undoqueue
            if undoCommand != None:
                undoCommandList.extend( undoCommand )

            # reset commands put commands into redoqueue
            aCommand.reset()
#            t+=[["reverse+reset", time.time() ]]
        if undoCommandList != []:
            self.theUndoQueue.push ( undoCommandList )
            self.theRedoQueue.push ( aCommandList )
            self.changesSaved = False
            
        self.theMainWindow.update()
#        t+=[["Mainwindow update", time.time() ] ]
        self.checkAutoSaveOption()
#        t+=[["autosave", time.time() ]]
#        t0=t[0][1]

#        for anitem in t:
#            print anitem[0], anitem[1]-t0
#            t0=anitem[1]
#        print
            

    def autoSave(self, aFileName = 'AutoSaveUntitled'):
        processId = os.getpid()
        #If user has loaded a file, theModelName shouldn't be empty
        if self.theModelName != '' or self.theModelName != None:
            modelBaseName = os.path.splitext(self.theModelName)[0]
            autoSaveName = str(os.getcwd()) + os.sep + modelBaseName +  ".sav.eml"
        #Default save aFileName
        else:
            autoSaveName = str(os.getcwd()) + os.sep + aFileName +  ".sav.eml"

        self.saveEmlAndLeml(autoSaveName)

        if self.theUpdateInterval != 0:
            self.theTimer = gtk.timeout_add(self.theUpdateInterval, self.autoSave)         
  
         
         
              
        
    def checkAutoSaveOption(self): #def OperationAutoSave

        self.theUpdateInterval = self.getAutosavePreferences()[0] *1000
        byOperation = self.getAutosavePreferences()[1]
               
        if self.theUpdateInterval != 0 and self.theTimer == None:
            self.autoSave(os.path.splitext(os.path.basename(self.theModelFileName))[0])
              
        if byOperation !=0 :        
            if self.operationCount == byOperation:
                self.operationCount=1
                self.autoSave()     
            elif self.operationCount < byOperation:
                self.operationCount = self.operationCount +1


        #self.autoSave(os.path.splitext(os.path.basename(self.theModelFileName))[0])
        



    def canUndo( self ):
        if self.theUndoQueue == None:
            return False
        return self.theUndoQueue.isPrevious()

    def canRedo( self ):
        if self.theRedoQueue == None:
            return False
        return self.theRedoQueue.isNext()


    def undoCommandList( self ):
        aCommandList = self.theUndoQueue.moveback()
        self.theRedoQueue.moveback()
        cmdList = aCommandList[:]
        cmdList.reverse()
        for aCommand in cmdList:
            # execute commands
            aCommand.execute()
            ( aType, anID ) = aCommand.getAffectedObject()
            self.updateWindows( aType, anID )
            ( aType, anID ) = aCommand.getSecondAffectedObject()
            if aType == None and anID == None:
                pass
            else:
                self.updateWindows( aType, anID )

            aCommand.reset()
        self.changesSaved = False


    def redoCommandList( self ):
        aCommandList = self.theRedoQueue.moveforward()
        self.theUndoQueue.moveforward()
        for aCommand in aCommandList:
            # execute commands
            #try:
            aCommand.execute()

            #except:
            #   self.printMessage( 'Error in operation.', ME_ERROR )
            #   anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
            #       sys.exc_traceback), '\n' )
            #   self.printMessage( anErrorMessage, ME_PLAINMESSAGE )

            ( aType, anID ) = aCommand.getAffectedObject()
            self.updateWindows( aType, anID )
            ( aType, anID ) = aCommand.getSecondAffectedObject()
            if aType == None and anID == None:
                pass
            else:
                self.updateWindows( aType, anID )

            aCommand.reset()
        self.changesSaved = False


    def createStepperWindow( self ):
        newWindow = StepperWindow(self,'top_frame' )  
        newWindow.openWindow()
        self.theMainWindow.attachTab( newWindow, "Stepper" )      
        self.theStepperWindowList.append( newWindow )

    def createEntityWindow( self ):
        newWindow = EntityListWindow( self ,'top_frame' )
        newWindow.openWindow()
        self.theMainWindow.attachTab( newWindow,"EntityList" ) 
        self.theEntityListWindowList.append( newWindow )


    def createPathwayEditor( self, aLayout ):
        newWindow = PathwayEditor( self, aLayout, 'top_frame' )
        newWindow.openWindow()
        self.theMainWindow.attachTab( newWindow, "Pathway" ) 
        self.thePathwayEditorList.append( newWindow )


    def toggleOpenLayoutWindow(self,isOpen):
        self.openLayoutWindow=isOpen

    def createLayoutWindow( self ):
        if not self.openLayoutWindow:
            newWindow = LayoutManagerWindow( self )
            self.theLayoutManagerWindow=newWindow
            newWindow.openWindow()
            self.openLayoutWindow=True
        else:       
            self.theLayoutManagerWindow.present()

    def createObjectEditorWindow(self, aLayoutName, anObjectID ):
        if not self.openObjectEditorWindow:
            ObjectEditorWindow(self, aLayoutName, anObjectID)
        else:
            self.theObjectEditorWindow.setDisplayObjectEditorWindow( aLayoutName, anObjectID)
        
    def toggleObjectEditorWindow(self,isOpen,anObjectEditor):
        self.theObjectEditorWindow=anObjectEditor
        self.openObjectEditorWindow=isOpen

    def deleteObjectEditorWindow( self ):
        self.theObjectEditorWindow = None

    def createConnObjectEditorWindow(self, aLayoutName, anObjectID ):
        if not self.openConnObjectEditorWindow:
            ConnectionObjectEditorWindow(self, aLayoutName, anObjectID)
        else:
            self.theConnObjectEditorWindow.setDisplayConnObjectEditorWindow( aLayoutName, anObjectID)
        
    def toggleConnObjectEditorWindow(self,isOpen,aConnObjectEditor):
        self.theConnObjectEditorWindow=aConnObjectEditor
        self.openConnObjectEditorWindow=isOpen


    def createAboutModelEditor(self):
        if not self.openAboutModelEditor:
            AboutModelEditor(self)

    def toggleAboutModelEditor(self,isOpen,anAboutModelEditorWindow):
        self.theAboutModelEditorWindow = anAboutModelEditorWindow
        self.openAboutModelEditor=isOpen
       

    def copy(self ):
        self.theLastComponent.copy()


    def cut( self ):
        self.theLastComponent.cut()


    def paste(self ):
        self.theLastComponent.paste()


    def getCopyBuffer( self ):
        return self.copyBuffer



    def setCopyBuffer( self, aBuffer ):
        self.copyBuffer = aBuffer



    def setLastUsedComponent( self, aComponent ):
        self.theLastComponent = aComponent
        self.theMainWindow.update()



    def getLastUsedComponent( self  ):
        return self.theLastComponent 


    
    def getADCPFlags( self ):
        if self.theLastComponent == None:
            return [ False, False, False, False ]
        else:
            if self.copyBuffer == None:
                aType = None
            else:
                aType = self.copyBuffer.getType()
                
            return self.theLastComponent.getADCPFlags( aType )

    def setFullIDBrowser( self, aBrowser):
        self.theFullIDBrowser = aBrowser


    def updateWindows( self, aType = None, anID = None ):
        
        # aType None means nothing to be updated
        for aStepperWindow in self.theStepperWindowList:
            # anID None means all for steppers
            aStepperWindow.update( aType, anID )

        for anEntityWindow in self.theEntityListWindowList:
            anEntityWindow.update( aType, anID )
        if self.theObjectEditorWindow!=None:
            self.theObjectEditorWindow.update(aType, anID)
        if self.theConnObjectEditorWindow!=None:
            self.theConnObjectEditorWindow.update(aType, anID)

        for aPathwayEditor in self.thePathwayEditorList:
            aPathwayEditor.update( aType, anID )
            #self.theMainWindow['combo1'].set_popdown_strings(['Choose...'] + self.theLayoutManager.getLayoutNameList())
            #self.theMainWindow['combo1'].entry.set_text('Choose...')

        if self.theFullIDBrowser != None:
            self.theFullIDBrowser.update( aType,anID )
        if self.theMainWindow.exists():
            self.theMainWindow.update()
        #self.theLayoutManager.update( aType, anID )
        if self.theLayoutManagerWindow!=None:
            self.theLayoutManagerWindow.update()
        

    def __closeModel( self ):
        """ 
        in: Nothing
        returns True if successful, False if not
        """
        if not self.changesSaved:
            if self.printMessage("Changes are not saved.\n Are you sure you want to close %s?"%self.theModelName,ME_OKCANCEL)  == ME_RESULT_CANCEL:

                return False

        # close ModelStore
        self.theModelStore = None
        self.theLayoutManager = None
        self.theMultiplexer = None

        # set name 
        self.theModelName = ''
        self.theUndoQueue = None
        self.theRedoQueue = None
        self.changesSaved = True
        self.theLayoutManager = None
        self.__closeWindows()
        #self.updateWindows()
        return True


    def __closeWindows( self ):
        for aStepperWindow in self.theStepperWindowList:
            # anID None means all for steppers
            aStepperWindow.close( )
        for anEntityWindow in self.theEntityListWindowList:
            anEntityWindow.close( )

        for aPathwayEditor in self.thePathwayEditorList:
            aPathwayEditor.close( )
        if self.theConnObjectEditorWindow!=None:
            self.theConnObjectEditorWindow.destroy()
        if self.theObjectEditorWindow!=None:
            self.theObjectEditorWindow.destroy()
        if self.theLayoutManagerWindow!=None:
            self.theLayoutManagerWindow.close()
            self.theLayoutManagerWindow=None
            self.toggleOpenLayoutWindow(False)


    def __createModel(self ):
        """ 
        in: nothing
        out nothing
        """
        self.__closeModel()
        # create new model
        self.theModelStore = ModelStore()

        self.theLayoutManager = LayoutManager( self )
        self.theMultiplexer = CommandMultiplexer( self, self.theLayoutManager )
        self.theUndoQueue = CommandQueue(MAX_REDOABLE_COMMAND)
        self.theRedoQueue = CommandQueue(MAX_REDOABLE_COMMAND)
        

        
        # initDM

        # set name
        self.theModelName = 'Untitled'
        self.modelHasName = False
        self.changesSaved = True
        self.openLayoutWindow = False
        self.openObjectEditorWindow = False
        self.openConnObjectEditorWindow = False
        self.theLastComponent = None

        # init Autosave variables
        self.operationCount = 0
        self.theUpdateInterval = 0
        if self.theTimer != None:
            gtk.timeout_remove(self.theTimer)
            self.theTimer = None
            




    def __loadRecentFileList( self ):
        """
        in: nothing
        returns nothing
        """
        self.__theRecentFileList = []
        aFileName = self.__getHomeDir() + os.sep +  RECENTFILELIST_FILENAME

        # if not exists create it
        if not os.path.isfile( aFileName ):
            self.__saveRecentFileList()
        else:
            # if exists open it
            aFileObject = open ( aFileName, 'r' )

            # parse lines
            aLine = aFileObject.readline()
            while aLine != '':
                self.__theRecentFileList.append( aLine.strip( '\n' ) )
                aLine = aFileObject.readline()
                
            # close file
            aFileObject.close()


    def __addToRecentFileList(self, aFileName ):
        """
        in: string aFileName (including directory, starting from root)
        returns nothing
        """

        # check wheter its already on list
        for aFile in self.__theRecentFileList:
            if aFile == aFileName:
                return      

        # insert new item as first
        self.__theRecentFileList.insert ( 0, aFileName )

        # if length is bigger than max delete last 
        if len(self.__theRecentFileList) > RECENTFILELIST_MAXFILES:
            self.__theRecentFileList.pop()

        # save to file whole list
        self.__saveRecentFileList()

    def __getTmpDir(self):
        tmpDirPath = tempfile.gettempdir()
        return tmpDirPath

    def __setEmlModelTmpFileName(self, setType = None, aFileName = None):
        """
        Function: For saving as eml in /tmp, used later for conversion to em/sbml
        in: setType
            1 = for saving as em file to target dir, appending of 'l'
            2 = for saving as sbml file to target dir, replace ext sbml to eml              
        
        """
        
        if aFileName != None:    
            tmpFileBaseName = os.path.basename(aFileName)
            tmpExt = os.path.splitext( tmpFileBaseName )[1]
            if setType == 1:
                tmpaFileName = str(self.__getTmpDir()) + os.sep + tmpFileBaseName + 'l'
                self.theModelTmpFileName = tmpaFileName
        else:
            return

    def __getHomeDir( self ):
        """
        in: nothing
        returns the Home directory of user
        workaround for a bug in Python os.path.expanduser in Windows
        """
        
        aHomeDir = os.path.expanduser( '~' )
        if aHomeDir not in ( '~', '%USERPROFILE%' ):
            return aHomeDir
        if os.name == 'nt' and aHomeDir == '%USERPROFILE%':
            return os.environ['USERPROFILE']


    def __saveRecentFileList( self ):
        """
        in: nothing
        returns nothing
        """
        # creates recentfiledir if does not exist
        aDirName = self.__getHomeDir() + os.sep + RECENTFILELIST_DIRNAME
        if not os.path.isdir( aDirName ):
            os.mkdir( aDirName )

        # creates file 
        aFileName = self.__getHomeDir() + os.sep + RECENTFILELIST_FILENAME
        aFileObject = open( aFileName, 'w' )
        for aLine in self.__theRecentFileList:
            # writes lines
            aFileObject.write( aLine + '\n' )

        # close file
        aFileObject.close()


    def __convertPropertyValueList( self, aValueList ):
        
        aList = list()

        for aValueListNode in aValueList:

            if type( aValueListNode ) in (type([]), type( () ) ):

                isList = False
                for aSubNode in aValueListNode:
                    if type(aSubNode) in (type([]), type( () ) ):
                        isList = True
                        break
                if isList:
                    aConvertedList = self.__convertPropertyValueList( aValueListNode )
                else:
                    aConvertedList = map(str, aValueListNode)

                aList.append( aConvertedList )
            else:
                aList.append( str (aValueListNode) )

        return aList




    def __getUniqueName( self, aType, aStringList ):
        """
        in: string aType, list of string aStringList
        returns: string newName     
        """
        i = 1
        while True:
            # tries to create 'aType'1
            newName = 'new' + aType + str(i)
            if not aStringList.__contains__( newName ):
                return newName
            # if it already exists in aStringList, creates aType2, etc..
            i += 1


    def __plainMessage( self, theMessage ):
        """
        in: string theMessage
        returns None
        """

        print theMessage



    def __loadStepper( self, anEml ):
        """stepper loader"""

        aStepperList = anEml.getStepperList()


        for aStepper in aStepperList:

            aClassName = anEml.getStepperClass( aStepper )
            self.theModelStore.createStepper( str( aClassName ),\
                str( aStepper ) )

            aPropertyList = anEml.getStepperPropertyList( aStepper )
            
            for aProperty in aPropertyList:

                aValue = anEml.getStepperProperty( aStepper, aProperty )
                self.theModelStore.loadStepperProperty( aStepper,\
                           aProperty,\
                           aValue )

                                             
    def __loadEntity( self, anEml, aSystemPath='/' ):

        aVariableList = anEml.getEntityList( DM_VARIABLE_CLASS, aSystemPath )
        aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
        aSubSystemList = anEml.getEntityList( 'System', aSystemPath )

        self.__loadEntityList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__loadEntityList( anEml, 'Process',  aSystemPath, aProcessList )
        self.__loadEntityList( anEml, 'System',   aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__loadEntity( anEml, aSubSystemPath )


    def __loadProperty( self, anEml, aSystemPath='' ):
        # the default of aSystemPath is empty because
        # unlike __loadEntity() this starts with the root system

        aVariableList  = anEml.getEntityList( 'Variable',  aSystemPath )
        aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
        aSubSystemList = anEml.getEntityList( 'System', aSystemPath )

        self.__loadPropertyList( anEml, 'Variable',\
                         aSystemPath, aVariableList )
        self.__loadPropertyList( anEml, 'Process',  aSystemPath, aProcessList )
        self.__loadPropertyList( anEml, 'System',\
                         aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__loadProperty( anEml, aSubSystemPath )

    def __loadPropertyList( self, anEml, anEntityTypeString,\
                        aSystemPath, anIDList ):

        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID

            aPropertyList = anEml.getEntityPropertyList( aFullID )

            aFullPNList = map( lambda x: aFullID + ':' + x, aPropertyList ) 
            aValueList = map( anEml.getEntityProperty, aFullPNList )
            map( self.theModelStore.loadEntityProperty,
                 aFullPNList, aValueList )

#####################################################################################################################   
    def __loadEntityList( self, anEml, anEntityTypeString,\
                          aSystemPath, anIDList ):
        
        aPrefix = anEntityTypeString + ':' + aSystemPath + ':'
        aMessage=''

        aFullIDList = map( lambda x: aPrefix + x, anIDList )
        aClassNameList = map( anEml.getEntityClass, aFullIDList )
        map( self.theModelStore.createEntity, aClassNameList, aFullIDList )
        """
        if len(self.theModelStore.getNotExistClass())>0:
            aMessage='There is no .desc file for '
            for aClass in self.theModelStore.getNotExistClass():
                aMessage=aMessage + aClass + ', '
            self.printMessage( aMessage[:-1], ME_PLAINMESSAGE )
            self.theModelStore.setNotExistClass()
        """


    def __saveStepper( self , anEml ):
        """stepper loader"""

        aStepperList = self.theModelStore.getStepperList()

        for aStepper in aStepperList:

            aClassName = self.theModelStore.getStepperClassName( aStepper )
            anEml.createStepper( str( aClassName ), str( aStepper ) )

            aPropertyList = self.theModelStore.getStepperPropertyList( aStepper )

            for aProperty in aPropertyList:
                
                anAttributeList = self.theModelStore.getStepperPropertyAttributes( aStepper, aProperty)

                # check get attribute 
                if anAttributeList[ME_SAVEABLE_FLAG]:
                                    
                    aValue = self.theModelStore.saveStepperProperty( aStepper, aProperty )
                    if aValue == '' or aValue == []:
                        continue

                    aValueList = list()
                    if type( aValue ) != type([]):
                        aValueList.append( str(aValue) )
                    else:
                        aValueList = self.__convertPropertyValueList( aValue )

                    anEml.setStepperProperty( aStepper, aProperty, aValueList )
    
    def __saveEntity( self, anEml, aSystemPath='/' ):

        aVariableList = self.theModelStore.getEntityList(  'Variable', aSystemPath )
        aProcessList   = self.theModelStore.getEntityList( 'Process', aSystemPath )
        aSubSystemList = self.theModelStore.getEntityList( 'System', aSystemPath )
        
        self.__saveEntityList( anEml, 'System',   aSystemPath, aSubSystemList )
        self.__saveEntityList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__saveEntityList( anEml, 'Process',  aSystemPath, aProcessList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveEntity( anEml, aSubSystemPath )

            
    def __saveEntityList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):

        for anID in anIDList:
           
            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aClassName = self.theModelStore.getEntityClassName( aFullID )

            if aClassName == 'System::/':
                pass
            else:
                anEml.createEntity( aClassName, aFullID )
            


    def __saveProperty( self, anEml, aSystemPath='' ):
        # the default of aSystemPath is empty because
        # unlike __loadEntity() this starts with the root system

        aVariableList  = self.theModelStore.getEntityList( 'Variable',\
                                                          aSystemPath )
        aProcessList   = self.theModelStore.getEntityList( 'Process',\
                                                          aSystemPath )
        aSubSystemList = self.theModelStore.getEntityList( 'System',\
                                                          aSystemPath )

        self.__savePropertyList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__savePropertyList( anEml, 'Process', aSystemPath, aProcessList )
        self.__savePropertyList( anEml, 'System', aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveProperty( anEml, aSubSystemPath )



    def __savePropertyList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):

        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aPropertyList = self.theModelStore.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:
                aFullPN = aFullID + ':' + aProperty
                
                anAttributeList = self.theModelStore.getEntityPropertyAttributes(aFullPN)

                # check savable
                if anAttributeList[ME_SAVEABLE_FLAG]:
                    
                    aValue = self.theModelStore.saveEntityProperty(aFullPN)

                    if aValue != '' and aValue != []:

                        aValueList = list()
                        if type( aValue ) != type([]):
                            aValueList.append( str(aValue) )

                        else:
                            # ValueList convert into string for eml
                            aValueList = self.__convertPropertyValueList( aValue )

                            
                        anEml.setEntityProperty( aFullID, aProperty, aValueList )




    
    def createAutosaveWindow(self, aDuration):
        anAutosaveWindow = AutosaveWindow.AutosaveWindow(self, aDuration)
        return anAutosaveWindow.return_result()
        
            
    def getAutosavePreferences( self ):
        AutosavePreferences = []
        AutosavePreferences.append(int(self.getParameter('duration')))
        AutosavePreferences.append(int(self.getParameter('operations')))
        return AutosavePreferences

    def setAutosavePreferences( self, AutosavePreferences ):
        """
        saves autosave preferences into config database
        """
        self.operationCount = 0
        self.theUpdateInterval = 0 
        if self.theTimer != None:
            gtk.timeout_remove( self.theTimer )
            self.theTimer = None
        self.setParameter('duration', AutosavePreferences[0])
        self.setParameter('operations',AutosavePreferences[1])
        self.saveParameters()

    def setParameter(self, aParameter, aValue):
        """tries to set a parameter in ConfigDB
        if the param is not present in either autosave or default section    raises exception and quits
        """
        # first try to set it in autosave section
        if self.theConfigDB.has_section('autosave'):
            if self.theConfigDB.has_option('autosave',aParameter):
                self.theConfigDB.set('autosave',aParameter, str(aValue))
        #else:

            # sets it in default
        #    self.theConfigDB.set('DEFAULT',aParameter, str(aValue))

    def saveParameters( self ):
        """tries to save all parameters into a config file in home directory
        """
        self.theIniFolderName = GUI_HOMEDIR + os.sep + '.modeleditor' + os.sep
        if os.path.isdir(self.theIniFolderName):
            fp = open( self.theIniFileName, 'w' )
            self.theConfigDB.write( fp )
        #except:
         #   self.message("Couldnot save preferences into file %s.\n Please check permissions for home directory.\n"%self.theIniFileName)
        else:
            os.spawnlp(os.P_WAIT,'mkdir','mkdir',self.theIniFolderName)
  
    def getParameter(self, aParameter):
        """tries to get a parameter from ConfigDB
        if the param is not present in either autosave or default section
        raises exception and quits
        """

        # first try to get it from autosave section
        if self.theConfigDB.has_section('autosave'):
            if self.theConfigDB.has_option('autosave',aParameter):
                return self.theConfigDB.get('autosave',aParameter)

        # gets it from default
        #return self.theConfigDB.get('DEFAULT',aParameter)


   
    def getParametersFromConfigParser(self):
        parameters = []
        parameters.append(self.theConfigDB.getint('autosave','duration'))
        parameters.append(self.theConfigDB.getint('autosave','operations'))
        return parameters   

    
   
    def openConfirmWindow(self,  aMessage, aTitle, isCancel = 1 ):
        """ pops up a modal dialog window
            with aTitle (str) as its title
            and displaying aMessage as its message
            and with an OK and a Cancel button
            returns:
            True if Ok button is pressed
            False if cancel button is pressed
        """
        aConfirmWindow = ConfirmWindow(isCancel, aMessage, aTitle )
        return aConfirmWindow.return_result() == OK_PRESSED

    def __readIni(self,aPath):
        """read preferences.ini file
        an preferences.ini file may be in the given path
        that have an autosave section or others but no default
        argument may be a filename as well
        """

        # first delete every section apart from default
        for aSection in self.theConfigDB.sections():
            self.theConfigDB.remove(aSection)

        # gets pathname
        if not os.path.isdir( aPath ):
            aPath=os.path.dirname( aPath )

        # checks whether file exists
        aFilename=aPath+os.sep+ '.modeleditor' + os.sep + 'preferences.ini'
        if not os.path.isfile( aFilename ):
            # self.message('There is no preferences.ini file in this directory.\n Falling back to system defauls.\n')
            return None

        # tries to read file

        try:
            self.message('Reading preferences.ini file from directory [%s]' %aPath)
            self.theConfigDB.read( aFilename )

        # catch exceptions
        except:
            self.message(' error while executing ini file [%s]' %aFileName)
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
            self.message(anErrorMessage)

