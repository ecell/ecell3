#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of E-CELL Model Editor package
#
#               Copyright (C) 1996-2003 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#
from gtk import *
import gtk.gdk
from ListWindow import *
import os
import os.path
import string
from Constants import *
from LayoutCommand import *
from PathwayEditor import * 
from ConfirmWindow import *
from Runtime import *
import ModelEditor
import webbrowser


        
class MEMainWindow( ListWindow ):

    
    def __init__( self, theModelEditor):
        
        """
        in: ModelEditor theModelEditor
        returns nothing
        """
        self.noOfStepper = 0
        self.noOfEntity = 0
        self.noOfClass = 0
        self.lockTabs = False
        self.lockEntry = False
        #self.noOfPathway = 0
        
       # init superclass

        ListWindow.__init__( self, theModelEditor )
        self.theModelEditor = theModelEditor
        self.theRuntimeObject = self.theModelEditor.theRuntimeObject
 


        

        
    def openWindow( self ):
        
        """
        in: nothing
        returns nothing
        """

        # superclass openwindow
        ListWindow.openWindow( self )
        self['ObjectWindow'].remove_page(0 )
        # add signal handlers
        
        self.addHandlers({ 
            
            'on_new_menu_activate' : self.__new_menu_clicked,\
            
            'load_model_menu_activate' : self.__load_menu_clicked,\
            'save_model_menu_activate' : self.__save_menu_clicked,\
            'on_save_model_as_activate' : self.__save_as_menu_clicked,\
            'on_validate_model_activate' : self.__validate_menu_clicked,\
            'on_import_menu_activate' : self.__import_menu_clicked,\
            'exit_menu_activate' : self.__quit_menu_clicked,\
            'on_undo_activate' : self.__undo_activated,\
            'on_redo_activate' : self.__redo_activated,\
            'stepper_window_activate' : self.__stepper_window_activated,\
            'entity_window_activate' : self.__entity_window_activated,\
            'on_pathway_editor_activate' : self.__pathway_editor_activated,\
            'about_menu_activate' : self.__about_menu_clicked,\
            'on_autosave_preferences1_activate' : self.__autosave_window_activated,\
            'on_how-to_activate' : self.__guide_activated,\
            'on_manual_activate' : self.__manual_activated,\
            'on_new_button_clicked' : self.__new_menu_clicked,\
            'on_open_button_clicked' : self.__load_menu_clicked,\
            'on_save_button_clicked' : self.__save_menu_clicked,\
            'on_quit_button_clicked' : self.__quit_menu_clicked,\
            'on_redo_button_clicked' : self.__redo_activated,\
            'on_undo_button_clicked' : self.__undo_activated,\
            'on_start_button_clicked' : self.__start_activated,\
            'on_stop_button_clicked' : self.__stop_activated,\
            'on_step_button_clicked' : self.__step_activated,\
            'on_StepperButton_clicked' : self.__stepper_window_activated,\
            'on_EntityButton_clicked' : self.__entity_window_activated,\
            'on_PathwayButton_clicked' : self.__pathway_editor_activated,\
            'on_LayoutButton_clicked' : self.__layout_window_activated,\
            'on_About_button_clicked' : self.__about_menu_clicked,\
            'on_scrolledwindow1_expose_event' : self.__scrollwindow_exposed,\
            'on_layout_name_entry_changed': self.__on_combo_entry_changed,\
            'on_ClassButton_clicked': self.__class_window_activated,\
            'on_design_mode_clicked' : self.__design_mode_clicked,\
            #'on_layout_name_entry_activate' : self.__rename_layout,
            'on_layout_name_entry_editing_done' : self.__rename_layout,\
            'on_run_mode_clicked' : self.__run_mode_clicked
            })
        self.noModeToggle = False
        self.comboEntryHandlerID = self['layout_name_entry'].connect('changed', self.__on_combo_entry_changed )
        self['ObjectWindow'].connect ( "switch-page", self.__changeNotebookTab )
        self['MEMainWindow'].connect("delete-event", self.deleted)     
        # set up message textbox buffer
        self.theSmallWindow = self['handlebox24']
        self.theMessageBuffer = gtk.TextBuffer(None)
        self['textview1'].set_buffer(self.theMessageBuffer)
        self.theMessageWindow = self['scrolledwindow1']
        self.smallWindowAttachment = self.theMessageWindow
        endIter=self.theMessageBuffer.get_end_iter()
        self.endMark=self.theMessageBuffer.create_mark('EM',endIter,gtk.FALSE)
        self.setIconList( os.environ['MEPATH'] + os.sep + "glade" +
                     os.sep + "modeleditor.png", os.environ['MEPATH'] + os.sep
                     + "glade" + os.sep + "modeleditor32.png" )
                             
        self['stop_button'].set_sensitive(gtk.FALSE)                           
        self['entry5'].set_text(str(0.0))
        self['entry4'].set_text(str(1))
        
        # update 
        self['MEMainWindow'].maximize()
        self.theStatusBar = self['statusbar']
        self.contextID = self.theStatusBar.get_context_id( "ME" )
        self.update()
        
    def printOnStatusbar( self, aText ):
        self.theStatusBar.push( self.contextID, aText )
        
    def setSmallWindow( self, aFrame ):
        if aFrame == self.smallWindowAttachment:
            return
        self.theSmallWindow.remove( self.smallWindowAttachment )
        self.smallWindowAttachment = aFrame
        self.theSmallWindow.add( aFrame )
        aFrame.show_all()
    
    def restoreMessageWindow( self ):
        self.setSmallWindow( self.theMessageWindow )
    
   
    def attachTab( self, aWindow, aType ):
        #if aWindow['top_frame'].parent == None:
        #    return
        
        topFrame = aWindow['top_frame']
        aContainer = gtk.HBox()
        if aType == "Pathway":
            aLabel = aWindow.getLabelWidget()
        else:
            aLabel = gtk.Label( aType )
        aContainer.pack_start( aLabel )
        aButton = gtk.Button()
        newImg = gtk.Image()
        anImg = self['imagex']
        aPixbuf = anImg.get_pixbuf()
        newPixbuf = aPixbuf.copy()
        newImg.set_from_pixbuf( aPixbuf)
        aButton.add(newImg )
        aContainer.pack_end( aButton )
        aButton.connect( "clicked", self.__closeTab, aWindow)
        self.lockTabs = True
        #error occurred here
        self['ObjectWindow'].append_page( topFrame, aContainer)
        aContainer.expand = gtk.FALSE
        aContainer.fill = gtk.FALSE
        aContainer.show_all()
        topFrame.show_all()
        topFrame.set_data("ParentWindow",aWindow)
        self.lockTabs = False
        self.presentTab(aWindow)
        self.theModelEditor.setupDNDDest( topFrame )
        self.restoreMessageWindow()
        
 
    def detachTab( self, aWindow ):
        pageNumber = self['ObjectWindow'].page_num( aWindow['top_frame'] )
        self.lockTabs = True
        self['ObjectWindow'].remove_page( pageNumber )
        self.lockTabs = False
        self.noOfClass=0
        self.restoreMessageWindow()
        self.update()
    
    def presentTab( self, aWindow):
        pageNumber = self['ObjectWindow'].page_num( aWindow['top_frame'])
        self.lockTabs = True
        self['ObjectWindow'].set_current_page( pageNumber )
        self.lockTabs = False
        self.update()


    def __closeTab( self, *args ):
        #args[1].close()
        args[1].deleted(None)

    

    def openModel( self ):

        """
        in: nothing
        returns nothing
        """
        aFileName = self.__getFileSelection( self.theModelEditor.loadDirName )
        
        if aFileName == None:
            return

        # call modeleditor loadModel
        self.theModelEditor.loadModel( aFileName )       



    def openRecentModel( self, aNumber ):
        """
        in: nothing
        returns nothing
        """

        # get filename and dirs
        recentList = self.theModelEditor.getRecentFileList()

        # call modeleditor loadModel
        self.theModelEditor.loadModel( recentList[ aNumber ] )


    def saveModel ( self, saveAsFlag = False ):
        """
        in: bool saveAsFlag
        returns nothing 
        """


        if saveAsFlag or not self.theModelEditor.modelHasName or not self.theModelEditor.isNameConfirmed:
            aFileName = self.__getFileSelection( self.theModelEditor.saveDirName )

            # Convert All FileExtensions to Lower Case here
            if aFileName != None and aFileName != '':
                aFileName = self.theModelEditor.filenameFormatter(aFileName)
                        
                if os.path.splitext(aFileName)[0] == '': 
                    self.theModelEditor.printMessage("No FileName specified", ME_WARNING) 
        else:
            aFileName = self.theModelEditor.theModelFileName

        if aFileName == None:
            return

        # call modeleditor savemodel

        if self.theModelEditor.changesSaved and aFileName == self.theModelEditor.theModelFileName and not self.theModelEditor.getMode() == ME_RUN_MODE:
            return aFileName
        self.theModelEditor.saveModel( aFileName )
        if self.theModelEditor.changesSaved:
            return aFileName

    def importModel ( self ):
        """
        in: nothing
        returns nothing
        """
        self.theModelEditor.printMessage("Sorry, not implemented !", ME_ERROR )


    def displayMessage ( self, aMessage ):
        """
        in: string or list of string aMessage
        returns nothing
        """
        self.restoreMessageWindow()
        iter = self.theMessageBuffer.get_iter_at_mark( self.endMark )
        # writes string or list of strings to end of buffer
        if type(aMessage) == list:  
            
            # If first string is not '\n', add it.
            if len(aMessage)>0:
                if string.find(aMessage[0],'\n') != 0:
                    aMessage[0] = '\n' + aMessage[0]

            for aLine in aMessage:
                aString = str( aLine )
                
        else: 
            aString = str( aMessage )
            if string.find(aString,'\n') != 0:
                aString = '\n' + aString
        self.theMessageBuffer.insert(  iter, aString , len(aString) )

        # scrolls textwindow to end of buffer
        self['textview1'].scroll_to_mark(self.endMark,0)


    def update ( self):
        """
        in: nothing
        returns nothing
        """

        # update View menu and buttons
        if not self.exists():
            return

        # update recent files menus
        recentList = self.theModelEditor.getRecentFileList()
        if len( recentList ) > 0:
            i = 1
            subMenu = gtk.Menu()
            for aFile in recentList:
        
                # get filename
                aFileName = os.path.split( aFile )[1]

                # add number
                aString = str(i) + ': ' + aFileName
        
                # create and add menuitem
                aMenuItem = gtk.MenuItem( aString )
                aMenuItem.connect ( 'activate', self.__recent_menu_clicked )
                aMenuItem.set_data( 'Number', i )
 
                subMenu.append( aMenuItem )
                i+=1

            # attach menu to menuitem
            subMenu.show_all()
            self['load_recent'].set_submenu( subMenu )

        # update title
        if self['MEMainWindow'] != None:
            if self.theModelEditor.theModelName != '' :
                aTitle = os.path.split ( self.theModelEditor.theModelName ) [1]
                self['MEMainWindow'].set_title( aTitle )

            else:
                self['MEMainWindow'].set_title( 'ModelEditor')

        # update undo, redo buttons, menus
        if self.theModelEditor.canUndo():
            undoFlag = gtk.TRUE
        else:
            undoFlag = gtk.FALSE

        self['button1'].set_sensitive( undoFlag )       
        self['undo1'].set_sensitive( undoFlag )

        if self.theModelEditor.canRedo():
            redoFlag = gtk.TRUE
        else:
            redoFlag = gtk.FALSE

        self['button2'].set_sensitive( redoFlag )       
        self['redo1'].set_sensitive( redoFlag )
        oldEntry = self['combo1'].entry.get_text()
        self.lockEntry = True
        popList = ['']
        if self.theModelEditor.theLayoutManager != None:
            popList = self.theModelEditor.theLayoutManager.getLayoutNameList()
        self['combo1'].set_popdown_strings( popList )
        self['combo1'].entry.set_text( oldEntry )            
        self.lockEntry = False
        #update Layout Manager components
        curpage = self['ObjectWindow'].get_current_page()
        topFrame = self['ObjectWindow'].get_nth_page( curpage )
        choosenLayout = "Choose..."
        if topFrame != None:
            aWindow = topFrame.get_data("ParentWindow")
            if aWindow.__class__.__name__ == 'PathwayEditor':
                choosenLayout = aWindow.getLayout().getName()

        if choosenLayout != oldEntry:
            self['combo1'].entry.set_text( choosenLayout )            
        

        # update copy, cut, paste buttons, menus
#       adcpFlags = self.theModelEditor.getADCPFlags()

#       if adcpFlags[ME_COPY_FLAG]:
#           copyFlag = gtk.TRUE
#       else:
#           copyFlag = gtk.FALSE

#       self['copy1'].set_sensitive( copyFlag )
#       self['button4'].set_sensitive( copyFlag )
#       if adcpFlags[ME_COPY_FLAG] and adcpFlags[ME_DELETE_FLAG]:
#           cutFlag = gtk.TRUE
#       else:
#           cutFlag = gtk.FALSE

#       self['cut1'].set_sensitive( cutFlag )
#       self['button3'].set_sensitive( cutFlag )

#       if adcpFlags[ME_PASTE_FLAG]:
#           pasteFlag = gtk.TRUE
#       else:
#           pasteFlag = gtk.FALSE

#       self['paste1'].set_sensitive( pasteFlag )
#       self['button5'].set_sensitive( pasteFlag )



    def displayHourglass ( self ):        
        gtkwindow = self['MEMainWindow']
        gdkwindow = gtkwindow.window
        cursor = gtk.gdk.Cursor( gtk.gdk.WATCH )
        gdkwindow.set_cursor( cursor )
        while gtk.events_pending():
            gtk.main_iteration_do()
        
    def resetCursor ( self ):
        gtkwindow = self['MEMainWindow']
        gdkwindow = gtkwindow.window
        cursor = gtk.gdk.Cursor( gtk.gdk.TOP_LEFT_ARROW )
        gdkwindow.set_cursor( cursor )
        while gtk.events_pending():
            gtk.main_iteration_do()

    def showAbout ( self ):
        # show about information
        self.theModelEditor.createAboutModelEditor()
        #self.theModelEditor.printMessage("Sorry, not implemented !", ME_ERROR )

   
    def __getFileSelection ( self, aDirname, aFileName = '' ):
        """
        in: str aDirname, str aFileName
        returns None if dir is selected, or cancel is pressed
        sets self.searchDirName if not cancel is pressed
        """
        defaultName = aDirname + os.sep + aFileName

        # create file selection dialog
        aDialog = gtk.FileSelection()

        # set init path for dialog
        aDialog.set_filename( defaultName )
        aPixbuf16 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                        os.sep + "glade" + os.sep + "modeleditor.png")
        aPixbuf32 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                        os.sep + "glade" + os.sep + "modeleditor32.png")
        aDialog.set_icon_list(aPixbuf16, aPixbuf32)
        aDialog.set_title("Select a file name")
        aDialog.show_fileop_buttons( )

        # make dialog modal
        aDialog.set_modal( gtk.TRUE )

        # present dialog       
        retVal = aDialog.run()

        aFileName = aDialog.get_filename()

        aDialog.hide()
        aDialog.destroy()
        
        if retVal == gtk.RESPONSE_CANCEL or retVal == gtk.RESPONSE_DELETE_EVENT:
            return None
        elif retVal == gtk.RESPONSE_OK and os.path.isdir(aFileName):

            self.theModelEditor.printMessage("Please Enter a valid filename", ME_WARNING)
            return None
            

        return aFileName

    def updateRunMode( self ):
        self.noModeToggle = True
        if self.theModelEditor.getMode() == ME_RUN_MODE:
            self['run_mode'].set_active( gtk.TRUE )
            self['design_mode'].set_active( gtk.FALSE )
        else:
            self['run_mode'].set_active( gtk.FALSE )
            self['design_mode'].set_active( gtk.TRUE )
        self.noModeToggle = False


    def checkToggleState( self, aMode ):
        self.theRuntimeObject.checkState( aMode )
        self.theModelEditor.updateWindows()
        

    #############################
    #      SIGNAL HANDLERS      #
    #############################
    def __design_mode_clicked( self, *args ):
        if self.noModeToggle:
            return
        if self['design_mode'].get_active():
            changeTo = ME_DESIGN_MODE
        else:
            changeTo = ME_RUN_MODE
        self.checkToggleState( changeTo )
    
    
    def __run_mode_clicked( self, *args ):
        if self.noModeToggle:
            return
        if self['run_mode'].get_active():
            changeTo = ME_RUN_MODE
        else:
            changeTo = ME_DESIGN_MODE
        self.checkToggleState( changeTo )


    def __gtk_button2_clicked(self,*arg):
        self.aClassWidget.destroy() 
        self.noOfClass = 0

    def __closeTab( self, *args ):
        #args[1].close()
        self.lockTabs = True
        args[1].deleted(None)
        self.lockTabs = False

    def __changeNotebookTab( self, *args ):
        if self.lockTabs == True:
            return gtk.TRUE
        else:

            topFrame = self['ObjectWindow'].get_nth_page(args[2])
            aWindow = topFrame.get_data("ParentWindow")

            if aWindow != None:

                self.presentTab(aWindow)
            self.restoreMessageWindow()
            return gtk.TRUE
   
        #self.update()
  
    def deleted( self, *arg ):
        self['layout_name_entry'].disconnect(self.comboEntryHandlerID )
        self.theModelEditor.quitApplication()
        return gtk.TRUE


    def __new_menu_clicked( self, *args ):
        self.theModelEditor.createNewModel()

    
    def __load_menu_clicked( self, *args ):
        self.openModel()
    
    def __save_menu_clicked( self, *args ):
        self.saveModel()
                      
    
    def __save_as_menu_clicked( self, *args ):
        self.saveModel( True )


    def __validate_menu_clicked( self, *args ):
        self.theModelEditor.validateModel()


    def __import_menu_clicked( self, *args ):
        self.importModel()


    def __recent_menu_clicked( self, *args ):
        if type( args[0] ) == gtk.MenuItem:
            aNumber = args[0].get_data( 'Number' )
            self.openRecentModel( aNumber -1 )


    def __quit_menu_clicked( self, *args ):
        if self.theModelEditor.isRunning():
            self.theModelEditor.theRuntimeObject.stop()

        self.theModelEditor.quitApplication()


    def __undo_activated(self, *args):

        self.theModelEditor.undoCommandList()


    def __redo_activated(self, *args):
        self.theModelEditor.redoCommandList()

    def __start_activated(self, *args) :        
        if self.theRuntimeObject.checkState(ME_RUN_MODE) == True:                    
            self.theRuntimeObject.run()
                          
  
    def __stop_activated(self, *args) :
        self.theRuntimeObject.stop()
            
    def updateRunPanel( self ):
        if self.theModelEditor.isRunning():
            self['stop_button'].set_sensitive(gtk.TRUE)
            self['start_button'].set_sensitive(gtk.FALSE)
            self['step_button'].set_sensitive(gtk.FALSE)
            self['entry5'].set_text(str(self.theRuntimeObject.getSimulationTime())[:10])       
        else:
            self['stop_button'].set_sensitive(gtk.FALSE)
            self['start_button'].set_sensitive(gtk.TRUE)
            self['step_button'].set_sensitive(gtk.TRUE)


    def __step_activated(self, *args): 
        text = self['entry4'].get_text()        
        

        if text == '' :
            dialog = ConfirmWindow(0,"Please enter a value as stepsize.")
            return

        try:
            aNum = float(text)
        except:
            dialog = ConfirmWindow(0,"Please enter a numeric value as step size")
            return            
        
        if aNum <= 0:                  
            dialog = ConfirmWindow(0,"Please enter a positive value as step size")
            return
         
        if self['radiobuttonSec'].get_active() == gtk.TRUE:
            if self.theRuntimeObject.checkState(ME_RUN_MODE) == True:
                self.theRuntimeObject.run(aNum)
                            
        elif self['radiobuttonStep'].get_active() == gtk.TRUE:                                               
            if self.theRuntimeObject.checkState(ME_RUN_MODE) == True:                                    
                self.theRuntimeObject.step(aNum)
                
                    

    def __stepper_window_activated( self, *args ):

        self.theModelEditor.createStepperWindow()
        


    def __entity_window_activated( self, *args ):

        self.theModelEditor.createEntityWindow()
       

    def __class_window_activated( self, *args ):

        self.theModelEditor.createClassWindow()


    def __autosave_window_activated( self, *args ):
       
        #get default autosave preferences
        
        aDuration = self.theModelEditor.getAutosavePreferences()
               
        newAutosaveWindow = self.theModelEditor.createAutosaveWindow(aDuration)
        
        if newAutosaveWindow != None:
            self.theModelEditor.setAutosavePreferences(newAutosaveWindow) 



    def __pathway_editor_activated( self, *args ):
        layoutManager = self.theModelEditor.theLayoutManager
        layoutName = layoutManager.getUniqueLayoutName()
        aCommand = CreateLayout( layoutManager, layoutName, True )  
        self.theModelEditor.doCommandList( [ aCommand ] )
        
            

        #self['combo1'].entry.set_text(layoutName)
        #self['combo1'].set_sensitive(gtk.TRUE)

    
    def __on_combo_entry_changed(self, *args):
        if self.lockEntry:
            return
        layoutName = self['combo1'].entry.get_text()    
        layoutManager = self.theModelEditor.theLayoutManager
        if layoutName != '' and layoutName != 'Choose...':
            aLayout = layoutManager.getLayout(layoutName) 
            if aLayout.isShown(): 
                self.presentTab(aLayout.getPathwayEditor())
            else:   
                layoutManager.showLayout(layoutName)
            

    def __layout_window_activated( self, *args ):

        self.theModelEditor.createLayoutWindow()


    def __about_menu_clicked( self, *args ):

        self.showAbout()


    def __autosave_window_activated( self, *args ):
       
        #get default autosave preferences
        
        aDuration = self.theModelEditor.getAutosavePreferences()
               
        newAutosaveWindow = self.theModelEditor.createAutosaveWindow(aDuration)
        
        if newAutosaveWindow != None:
            self.theModelEditor.setAutosavePreferences(newAutosaveWindow) 

    def __guide_activated(self, *args):
        webbrowser.open_new('file://'+ os.environ['MEPATH'] + os.sep + 'doc' + os.sep + 'HOW-TO.htm')

    def __manual_activated(self, *args):
        webbrowser.open_new('file://' + os.environ['MEPATH'] + os.sep + 'doc' + os.sep + 'Tutorial.htm')


    def __scrollwindow_exposed( self, *args ):

        pass

       

        
    def __rename_layout( self, *args ):
        print "editing done", args
 
