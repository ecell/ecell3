#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

# runtime.py should code for all the interactions btw 
#the GtkSessionMonitor.py and modeleditor.py


import tempfile
from ecell.gui_config import *
from ModelEditor import *
from ConfirmWindow import *

from Constants import *
import os
import sys
import traceback
import ecell.GtkSessionMonitor
import ecell.ecssupport
import time

if os.name == 'nt':
    try:
        import win32api
    except:
        pass
    ecell3_session_filename = "ecell3-session.exe"
else:
    ecell3_session_filename = "ecell3-session"

ecell3_session_path = os.environ["ECELL3_PREFIX"] + os.sep + "bin" + os.sep + ecell3_session_filename
TIME_THRESHOLD = 5

class Runtime:

    def __init__( self, aModelEditor ):
        #here initialize model
        self.theMode = ME_DESIGN_MODE
        self.theSession = None
        self.theModelEditor = aModelEditor
        self.testModels = []

    def closeModel( self ):
        self.changeToDesignMode( True )
        
    def createNewProcess( self, testPyFile ):
        if os.name == 'nt':
            return os.spawnl( os.P_NOWAIT, ecell3_session_path, '"' + ecell3_session_path + '"', '"' + testPyFile + '"' )
        else:
            return os.spawnl( os.P_NOWAIT, ecell3_session_path,  ecell3_session_path ,  testPyFile )
        
    def killProcess ( self, processID ):
        if os.name == 'nt':
            try:
                win32api.TerminateProcess( processID,0 )
            except:
                pass
        else:
            os.kill( processID,9 )
        
    def __preRun( self ):
        # THIS FUNCTION IS LINUX DEPENDENT!!!!
        # a possible workaround for using it under Windows is
        # use os.spawnl instead of os.spawnlp - the concrete path for ecell3-session must be found!
        # use import winapi32, winapi32.TerminateProcess( pid) instead of os.kill
        
        #attempting loading and executing model on a separate thread and check whether it is wise to run it in this thread
        if self.preRunSuccessful:
            return True
        tempDir = os.getcwd()
        processID = str(os.getpid() )
        fileName = self.theModelEditor.autoSaveName
        testPyFile = tempDir + os.sep + "test" + processID + ".py"
        testOutFile = tempDir + os.sep + "test" + processID + ".out"
        testFile = "\
list=[None, None]\n\
delay=10\n\
import gtk\n\
fd=open(r'" + testOutFile + "','w');fd.write('started'+chr(10));fd.flush()\n\
def timerhandler():\n\
    fd.write('a');fd.flush()\n\
    list[0]=gtk.timeout_add(delay,timerhandler)\n\
def mainfunction():\n\
    list[0]=gtk.timeout_add(delay, timerhandler)\n\
    try:\n\
        loadModel(r'" + fileName + "')\n\
    except:\n\
        pass\n\
    gtk.timeout_remove(list[0])\n\
    fd.write(chr(10)+'loaded'+chr(10));fd.flush()\n\
    try:\n\
        #self.theSimulator.initialize()\n\
        step(3)\n\
    except:\n\
        pass\n\
    fd.write('finished'+chr(10));fd.close()\n\
    gtk.main_quit()\n\
gtk.timeout_add( 10, mainfunction )\n\
gtk.main()\n\
"
        fd=open( testPyFile,'w' )
        fd.write( testFile )
        fd.close()
        fd=open( testOutFile, 'w' )
        fd.close()
        pid = self.createNewProcess( testPyFile )
        # first get started signal
        startedTime = time.time()
        startstr = ''
        a=[]
        while len(a)<1 and (time.time()-startedTime )< TIME_THRESHOLD:
            try:
                fd=open( testOutFile,'r' )
                a=fd.readlines(  )
                fd.close()
            except:
                pass
            if len(a)>0:
                startstr=a[0].strip()
        if startstr!="started":
            try:
                os.remove( testOutFile )
                os.remove( testPyFile )
            except:
                pass
            return False
            
        # must load it!!!
        lastReadTime = time.time()
        lastLength = 0
        while len(a)<3 and lastReadTime-time.time()<TIME_THRESHOLD:
            fd=open( testOutFile,'r' )
            a=fd.readlines(  )
            fd.close()
            if len(a)>1:
                newLength=len(a[1].strip())
                if newLength > lastLength:
                    lastLength = newLength
                    lastReadTime = time.time()
        if len(a) > 2:
            self.preLoadSuccessful = True
        else:
            try:
                os.remove( testOutFile )
                os.remove( testPyFile )
            except:
                pass

            self.killProcess( pid)            
            return False
                
        loadedTime = time.time()
        while startstr!='finished' and (time.time()-loadedTime )< TIME_THRESHOLD:
            fd=open( testOutFile,'r' )
            a=fd.readlines(  )
            fd.close()
            if len(a)>3:
                startstr=a[3].strip()
        try:
            os.remove( testOutFile )
            os.remove( testPyFile )
        except:
            pass


        if len(a)<4:
            # hanging or segfault
            self.killProcess( pid )

            return False
        else:
            self.preRunSuccessful = True
            return True

    #====================================================================================        
                
    def changeToRunMode( self ):
        self.preRunSuccessful= False
        self.preLoadSuccessful = False
        # DO VALIDATION first!!!

        # destroy BoardWindow, save tracerlist
        if self.theModelEditor.theResultsWindow != None:
            self.theModelEditor.theResultsWindow.closeBoardWindow()
        # destroy previous SessionMonitor:
        if self.theSession != None:
            # deregister previous callbacks
            self.theSession = None


        #save model to temp file

        if not self.theModelEditor.autoSave():
            return
        
        fileName = self.theModelEditor.autoSaveName
        self.theModelEditor.theMainWindow.displayHourglass()
        #instantiate GtkSessionMonitor 
        self.theSession = ecell.GtkSessionMonitor.GtkSessionMonitor()
        self.theSession.setMessageMethod( self.message )
        self.theSession.registerUpdateCallback( self.updateWindows )
        #load model into GtkSessionMonitor
        self.__preRun()
        if not self.preLoadSuccessful:
            dialog = ConfirmWindow(1,"This operation needs loading model into Simulator, but your model is not stable!\nTest load hanged or crashed! Are you sure you want to load model into Simulator?\nIf you choose yes, ModelEditor can hang or crash.\n(Changes are saved)", "CRITICAL" )
            if dialog.return_result() != 0:
                self.theModelEditor.theMainWindow.resetCursor()
                return False
            
        try:
            self.theSession.loadModel( fileName )                 
            #self.theSession.theSimulator.initialize()
        except:
            self.message(' Error while trying to parse model into simulator \n')
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
            self.message(anErrorMessage)
            self.theModelEditor.theMainWindow.resetCursor()
            dialog = ConfirmWindow(0,"Sorry, error in parsing model, see message window for details." )
            return False

        self.theMode = ME_RUN_MODE
        if self.theModelEditor.theResultsWindow != None:
            self.theModelEditor.theResultsWindow.openBoardWindow(self.theSession)
        
        self.theModelEditor.theMainWindow.resetCursor()
        self.theModelEditor.theMainWindow.updateRunMode()
        return True            
       

    def message( self, text):
        if type(text) == type(""): 
            if (text.find("Start") or text.find("Stop") ) and text.strip().find("\n") == -1 :
                self.theModelEditor.printMessage( text, ME_STATUSBAR )
                return          
        self.theModelEditor.printMessage( text )
     
    def isRunning( self ):
        if self.theSession == None:
            return False
        return self.theSession.isRunning()
    #====================================================================================                 
    def changeToDesignMode( self, forced = False ):
        # check whether simulation is running
        if self.isRunning():
            if not forced:
                dialog = ConfirmWindow(1, "To perform this operation, simulation must be stopped.\n Can simulation be stopped?")
                if dialog.return_result() == 0:
                    self.theSession.stop()
                else:
                    return False
            else:
                self.theSession.stop()
        self.theMode = ME_DESIGN_MODE
        self.theModelEditor.theMainWindow.updateRunMode()
        return True
        
    #====================================================================================             

    def createTracerWindow( self,fullPNStringList ):

        if self.theModelEditor.theResultsWindow == None:
            self.theModelEditor.createResultsWindow()

        #pass in number of columns
        self.theModelEditor.theResultsWindow.createTracer( fullPNStringList )
        
    #==================================================================================== 
    def attachToTracerWindow( self, aTitle, fullPNStringList ):
        
        self.theModelEditor.theResultsWindow.attachToTracerWindow( aTitle, fullPNStringList )

    
    #==================================================================================== 

    def getTracerList( self ):
        if self.theModelEditor.theResultsWindow != None:
            return self.theModelEditor.theResultsWindow.getTracerList()
        else:
            return []
        
    #==================================================================================== 

    def __canRun( self ):
        if not self.preRunSuccessful :
            dialog = ConfirmWindow(1,"Your model is not stable for runnig! Test run hanged or crashed!\nAre you sure you want to run in ModelEditor?\nIf you choose yes, ModelEditor can hang or crash.\n(Changes are saved)", "CRITICAL" )
            if dialog.return_result() != 0:
                return False
        return True



    #==================================================================================== 

    def run(self , aTime = 0 ):
            #do run test
        if self.__canRun():
            self.theSession.run( aTime )
        
    #====================================================================================     

    def stop( self ):            
        self.theSession.stop()
        
    #====================================================================================     

    def step( self, aNum ): 
        if self.__canRun():
            self.theSession.step( aNum )
            self.updateWindows()
        
        
    #====================================================================================              
    
    def getSession( self ):
        return self.theSession

    #====================================================================================     

    def checkState (self, passInState):             
        if self.theMode != passInState:                 
            if passInState == ME_DESIGN_MODE:
                return self.changeToDesignMode()
            else:
                return self.changeToRunMode()
        return True

    #====================================================================================     
    def updateWindows( self):
        self.theModelEditor.theMainWindow.updateRunPanel()
        # update Propertylists
        for aStepperWindow in self.theModelEditor.theStepperWindowList:
            # anID None means all for steppers
            aStepperWindow.update( None, None )

        for anEntityWindow in self.theModelEditor.theEntityListWindowList:
            anEntityWindow.update( None, None)           
        if self.theModelEditor.theObjectEditorWindow!=None:
            self.theModelEditor.theObjectEditorWindow.update(None, None)


    #====================================================================================     
            
    def getSimulationTime( self ):
        if self.theSession == None:
            return 0
        return self.theSession.getCurrentTime()  

    #====================================================================================     

    def createTracerSubmenu( self, aFullPNStringList ):
        subMenu = gtk.Menu()
        newMenuItem = gtk.MenuItem( "New Tracer" )
        newMenuItem.connect( 'activate', self.__subMenuHandler, ["newtracer",aFullPNStringList] )
        if len( aFullPNStringList) == 0:
            newMenuItem.set_sensitive( False )
        subMenu.append( newMenuItem )
        for aTracer in self.getTracerList():
            tracerMenu = gtk.MenuItem( aTracer )
            tracerMenu.connect( 'activate', self.__subMenuHandler, [ aTracer, aFullPNStringList] )
            if len( aFullPNStringList) == 0:
                tracerMenu.set_sensitive( False )
            subMenu.append( tracerMenu )
        subMenuItem = gtk.MenuItem( "Add to" )
        subMenuItem.set_submenu( subMenu )
        return subMenuItem
    
    #====================================================================================     

    def __subMenuHandler( self, menuItem, userData ):
        if not self.checkState( ME_RUN_MODE ):
            return
        if userData[0] == "newtracer":
            self.createTracerWindow( userData[1] )
        else:
            self.attachToTracerWindow( userData[0], userData[1] )
    


class ResultsWindow:
    # a wrapper for Osogo's boardwindow
    
    def __init__( self, aSession, aModelEditor ):
        self.theSession = aSession
        self.thePluginList = {}
        self.theBoardWindow = None
        self.theModelEditor = aModelEditor
        self.theStack = None
        self.theTopFrame = gtk.HBox()
        self.openBoardWindow(aSession)
        
    def openBoardWindow( self, aSession ):
        self.theSession = aSession
        self.theBoardWindow = self.theSession.openWindow("BoardWindow",'top_frame', self.theModelEditor.theMainWindow ) 
        self.theTopFrame.add( self.theBoardWindow['top_frame'] )
        self.theBoardWindow.setPackDirectionForward(True)
        self.theBoardWindow.setTableSize(2)
        if self.theStack != None:
            for anEntry in self.theStack.values():
                fullPNList = anEntry[:]
                self.createTracer( fullPNList )
            self.theStack = None
    
    
    def closeBoardWindow( self ):
        if self.theBoardWindow != None:
            self.theStack = {}
            for aPlugin in self.thePluginList.values():
                fullPNList = aPlugin.getRawFullPNList()[:]
                fullPNStringList = []
                for afullPN in fullPNList:
                    fullPNStringList.append(ecell.ecssupport.createFullPNString(afullPN ) )
                self.theStack[aPlugin.getTitle()] = fullPNStringList
            self.thePluginList = {}
            self.theTopFrame.remove(self.theBoardWindow['top_frame'])
            self.theBoardWindow.close()
            self.theBoardWindow = None
            
    
    def getBoardWindow( self ):
        return self.theBoardWindow()
    
    
    def deleted(self, *args ):
        self.thePluginList = {}
        self.closeBoardWindow()
        self.theModelEditor.theResultsWindow = None
        self.theTopFrame.destroy()
    
    def __getitem__( self, aWidgetName ):
        if aWidgetName == "top_frame":
            return self.theTopFrame
        return self.theBoardWindow[ aWidgetName ]
    
    
    
    def createTracer( self, aFullPNStringList ):
        aFullPNList = []
        for aFullPNString in aFullPNStringList:
            aFullPNList += [ ecell.ecssupport.createFullPN( aFullPNString ) ]
        aPlugin = self.theSession.createPluginOnBoard( 'TracerWindow', aFullPNList ) 
        self.thePluginList[ aPlugin.getTitle()] = aPlugin


    def attachToTracerWindow( self, aTitle, fullPNStringList ):
        
        aFullPNList = []
        for aFullPNString in fullPNStringList:
            aFullPNList += [ ecell.ecssupport.createFullPN( aFullPNString ) ]
        aPlugin = self.thePluginList[ aTitle ]
        aPlugin.appendRawFullPNList( aFullPNList )


    def getTracerList( self ):
        return self.thePluginList.keys()
        
         
    def getTracer( self, aTitle ):
        return self.thePluginList[aTitle]
