#!/usr/bin/env python

#from string import *
from OsogoPluginWindow import *

# column index of clist
PROPERTY_COL  = 0
VALUE_COL     = 1
GETABLE_COL   = 2
SETTABLE_COL  = 3
EDITABLE_COL  = 4

import gobject
import gtk
import os

PROPERTY_COL_TYPE=gobject.TYPE_STRING
VALUE_COL_TYPE=gobject.TYPE_STRING
GETABLE_COL_TYPE=gobject.TYPE_BOOLEAN
SETTABLE_COL_TYPE=gobject.TYPE_BOOLEAN

DISCARD_LIST=[ 'Name', 'Priority', 'StepperID', 'IsContinuous' ]


class PropertyWindow(OsogoPluginWindow):

    # ---------------------------------------------------------------
    # constructor
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __init__( self, aDirName, aData, aPluginManager, rootWidget=None ):

        # calls superclass's constructor
        OsogoPluginWindow.__init__( self, aDirName, aData,
                                   aPluginManager, rootWidget=rootWidget )
        self.theStatusBarWidget = None
        self.theParent = None
    # end of __init__

    def setParent ( self, aParent ):
        self.theParent = aParent


    def openWindow( self ):
        #self.openWindow()
        OsogoPluginWindow.openWindow(self)
        
        # add handers
        self.addHandlers( { 'on_checkViewAll_toggled' : self.updateViewAllProperties } )

        # initializes buffer
        self.thePreFullID = None
        self.thePrePropertyMap = {}

        # initializes ListStore
        self.theListStore=gtk.ListStore(
                        PROPERTY_COL_TYPE,
                        VALUE_COL_TYPE,
                        GETABLE_COL_TYPE,
                        SETTABLE_COL_TYPE )

        self['theTreeView'].set_model(self.theListStore)
        
        renderer=gtk.CellRendererText()
        column=gtk.TreeViewColumn( "Property", renderer, text=PROPERTY_COL)
        column.set_visible( gtk.TRUE )
        column.set_resizable(gtk.TRUE)
        self['theTreeView'].append_column(column)

        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__valueEdited)
        column=gtk.TreeViewColumn( "Value", renderer, text=VALUE_COL,
                                  editable=SETTABLE_COL )
        column.set_visible( gtk.TRUE )
        column.set_sizing( 1 ) # auto sizing
        self['theTreeView'].append_column(column)
        self.theValueColumn = column

        renderer=gtk.CellRendererToggle()
        column=gtk.TreeViewColumn("Get",renderer, active=GETABLE_COL )
        column.set_visible( gtk.TRUE )
        column.set_resizable(gtk.TRUE)
        self['theTreeView'].append_column(column)
        
        renderer=gtk.CellRendererToggle()
        column=gtk.TreeViewColumn("Set",renderer, active=SETTABLE_COL )
        column.set_visible( gtk.TRUE )
        column.set_resizable(gtk.TRUE)
        self['theTreeView'].append_column(column)

        # creates popu menu
        self.thePopupMenu = PropertyWindowPopupMenu(
                                        self.thePluginManager, self )
        # initializes statusbar
        self.theStatusBarWidget = self['statusbar']

        # set notebook page to Property tab
        self['notebookProperty'].set_current_page( 1 )

        # set default as not to view all properties
        self['checkViewAll'].set_active( False )
        self.theDiscardList = DISCARD_LIST

        if self.theRawFullPNList == ():
            return
        
        self.setIconList(
            os.environ['OSOGOPATH'] + os.sep + "ecell.png",
            os.environ['OSOGOPATH'] + os.sep + "ecell32.png")
        #self.__setFullPNList()
        self.update(True)

        #if ( len( self.theFullPNList() ) > 1 ) and ( aRoot != 'top_vbox' ):
        if ( len( self.theFullPNList() ) > 1 ) and ( rootWidget !=
                                                    'EntityWindow' ):
            self.thePreFullID = self.theFullID()
            aClassName = self.__class__.__name__

        # registers myself to PluginManager
        self.thePluginManager.appendInstance( self ) 





    # =====================================================================
    def setStatusBar( self, aStatusBarWidget ):
        """sets a status bar to this window. 
        This method is used when this window is displayed on other window.
        aStatusBarWidget  --  a status bar (gtk.StatusBar)
        Returns None
        [Note]:The type of aStatusBarWidget is wrong, throws exception.
        """

        if type(aStatusBarWidget) != gtk.Statusbar:
            raise TypeError("%s must be gtk.StatusBar.")

        self.theStatusBarWidget = aStatusBarWidget


    # =====================================================================
    def clearStatusBar( self ):
        """clear messaeg of statusbar
        """

        self.theStatusBarWidget.push(1,'')


    # =====================================================================
    def showMessageOnStatusBar( self, aMessage ):
        """show messaegs on statusbar
        aMessage   --  a message to be displayed on statusbar (str)
        [Note]:message on statusbar should be 1 line. If the line aMessage is
               more than 2 lines, connects them as one line.
        """

        aMessage = string.join( string.split(aMessage,'\n'), ', ' )

        self.theStatusBarWidget.push(1,aMessage)


    # ---------------------------------------------------------------
    # Overwrite Window.__getitem__
    # When this instance is on EntityListWindow,
    # self['statusbar'] returns the statusbar of EntityListWindow
    #
    # aKey  : a key to access widget
    # return -> a widget
    # ---------------------------------------------------------------
    def __getitem__( self, aKey ):

        # When key is not statusbar, return default widget
        if aKey != 'statusbar':
                return self.widgets.get_widget( aKey )

        # When is 'statusbar' and self.setStatusBar method has already called,
        # returns, the statusbar of EntityWindow.
        else:
            if self.theStatusBarWidget != None:
                return self.theStatusBarWidget
            else:
                return None

    # end of __getitem__

    # ---------------------------------------------------------------
    # Overwrite Window.setRawFullPNList
    # This method is used by EntityListWindow
    # change RawFullPNList
    #
    # aRawFullPNList  : a RawFullPNList
    # return -> None
    # ---------------------------------------------------------------
    def setRawFullPNList( self, aRawFullPNList ):

        # When aRawFullPNList is not changed, does nothing.
        if self.theRawFullPNList == aRawFullPNList:
            # do nothing
            pass

        # When aRawFullPNList is changed, updates its and call self.update().
        else:
            # update RawFullPNList
            OsogoPluginWindow.setRawFullPNList(self,aRawFullPNList)
            self.update()
       
    # end of setRawFullPNList


    # ---------------------------------------------------------------
    # update (overwrite the method of superclass)
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def update( self, fullUpdate = False ):


        # ----------------------------------------------------
        # checks a value is changed or not
        # ----------------------------------------------------

        # check the fullID
        if self.thePreFullID != self.theFullID():
            fullUpdate = True
            
        # check all property's value
        #if aChangedFlag == FALSE:
        #    anEntityStub = EntityStub( self.theSession.theSimulator, \
        #                   createFullIDString(self.theFullID()) )
        #
        #    for aProperty in anEntityStub.getPropertyList():
                # When a value is changed, 
        #        if self.thePrePropertyMap[aProperty] != anEntityStub.getProperty(aProperty):
        #            aChangedFlag = TRUE
        #            break

        # ----------------------------------------------------
        # updates widgets
        # ----------------------------------------------------
        # creates EntityStub
        anEntityStub = EntityStub( self.theSession.theSimulator, createFullIDString(self.theFullID()) )

        if fullUpdate == False:
            # gets propery values for thePreProperyMap in case value is not tuple
            for aPropertyName in self.thePrePropertyMap.keys():
                aProperty = self.thePrePropertyMap[aPropertyName]
                if type( aProperty[0] ) not in ( type( () ), type( [] ) ):
                    aProperty[0] = anEntityStub.getProperty(aPropertyName)
                
        else:

            self.theSelectedFullPN = ''

            # -----------------------------------------------
            # updates each widget
            # Type, ID, Path, Classname
            # -----------------------------------------------
            anEntityType = ENTITYTYPE_STRING_LIST[self.theFullID()[TYPE]]
            anID = self.theFullID()[ID]
            aSystemPath = str( self.theFullID()[SYSTEMPATH] )
            
            self['labelEntityType'].set_text( anEntityType + ' Property' )
            self['entry_classname'].set_text( anEntityStub.getClassname() )
            self['entry_id'].set_text( anID )
            self['entry_path'].set_text( aSystemPath  )
            
            if aSystemPath != '/' and anID != '/':
                anID = '/' + anID
            self['entryFullID'].set_text( aSystemPath + anID )

            # saves properties to buffer
            self.thePrePropertyMap = {}
            for aProperty in anEntityStub.getPropertyList():
                self.thePrePropertyMap[str(aProperty)] = [None, None]
                self.thePrePropertyMap[str(aProperty)][0] =\
                        anEntityStub.getProperty(aProperty)
                self.thePrePropertyMap[str(aProperty)][1] =\
                        anEntityStub.getPropertyAttributes(aProperty)
                
            # updates PropertyListStore
            self.__updatePropertyList()

        # save current full id to previous full id.
        self.preFullID = self.theFullID()

        # updates status bar
        if self['statusbar'] != None:
            self['statusbar'].push(1,'')



    def __valueEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """
        
        aNewValue = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        aSelectedProperty = self.theListStore.get_value( anIter, PROPERTY_COL )
        self.theSelectedFullPN = convertFullIDToFullPN( self.theFullID(),
                                                       aSelectedProperty )
        self.__updateValue( aNewValue, anIter, VALUE_COL )

    
    # ---------------------------------------------------------------
    # __updatePropertyList
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __updatePropertyList( self ):

        self.theList = []
        aPropertyList = self.thePrePropertyMap.keys()

        # do nothing for following properties
        try:
            aPropertyList.remove( 'FullID' )
            aPropertyList.remove( 'Name' )
        except:
            pass

        for aPropertyName in aPropertyList: # for (1)
            if aPropertyName not in self.theDiscardList:
                aProperty = self.thePrePropertyMap[aPropertyName]
                anAttribute = aProperty[1]

                # When the getable attribute is false, value is ''
                if anAttribute[GETABLE] == FALSE:
                    aValue = ''
                else:
                    aValue = str( aProperty[0] )

                aValueString = str( aValue )
                aList = [ aPropertyName, aValueString, anAttribute[GETABLE],
                         anAttribute[SETTABLE] ]
                self.theList.append( aList )

        self.theListStore.clear()

        for aValue in self.theList:
            iter=self.theListStore.append( )
            cntr=0
            for valueitem in aValue:
                self.theListStore.set_value(iter,cntr,valueitem)
                cntr+=1

    # end of __updatePropertyList


    # ---------------------------------------------------------------
    # updateValue
    #   - sets inputted value to the simulator
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __updateValue( self, aValue, anIter, aColumn ):


        # ------------------------------------
        # gets getable status
        # ------------------------------------
        aGetable = self.theListStore.get_value( anIter, GETABLE_COL )

        # ------------------------------------
        # checks the type of inputted value 
        # ------------------------------------
        if aGetable == TRUE:
            aPreValue = self.theListStore.get_value( anIter, aColumn )

            # ------------------------------------
            # when type is integer
            # ------------------------------------
            if type(aPreValue) == type(0):
                try:
                    aValue = string.atoi(aValue)
                except:
                    # print out traceback
                    import sys
                    import traceback
                    anErrorMessage = string.join(
                        traceback.format_exception( 
                            sys.exc_type,sys.exc_value,sys.exc_traceback ),
                                                 '\n' )
                    self.theSession.message("-----An error happens.-----")
                    self.theSession.message(anErrorMessage)
                    self.theSession.message("---------------------------")

                    # creates and display error message dialog.
                    anErrorMessage = "Input an integer!"
                    anErrorTitle = "The type error!"
                    if self['statusbar'] != None:
                        self['statusbar'].push(1,anErrorMessage)
                    anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
                    return None

            # ------------------------------------
            # when type is float
            # ------------------------------------
            elif type(aPreValue) == type(0.0):
                try:
                    aValue = string.atof(aValue)
                except:
                    # print out traceback
                    import sys
                    import traceback
                    anErrorMessage = string.join( traceback.format_exception( \
                        sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
                    self.theSession.message("-----An error happened.-----")
                    self.theSession.message(anErrorMessage)
                    self.theSession.message("---------------------------")

                    # creates and display error message dialog.
                    anErrorMessage = "Input a float!"
                    anErrorTitle = "The type error!"
                    if self['statusbar'] != None:
                        self['statusbar'].push(1,anErrorMessage)
                    anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
                    return None

            # ------------------------------------
            # when type is tuple
            # ------------------------------------
            elif type(aPreValue) == type(()):
                try:
                    aValue = convertStringToTuple( aValue )

                except:
                    # print out traceback
                    import sys
                    import traceback
                    anErrorMessage = string.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
                    self.theSession.message("-----An error happens.-----")
                    self.theSession.message(anErrorMessage)
                    self.theSession.message("---------------------------")

                    # creates and display error message dialog.
                    anErrorMessage = "Input a tuple!"
                    anErrorTitle = "The type error!"
                    if self['statusbar'] != None:
                        self['statusbar'].push(1,anErrorMessage)
                    anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
                    return None


        aFullPNString = createFullPNString(self.theSelectedFullPN)

        try:
            self.setValue( self.theSelectedFullPN, aValue ) 
            self['theTreeView'].get_selection().select_iter( anIter )
        except:

            # print out traceback
            import sys
            import traceback
            anErrorMessage = string.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
            self.theSession.message("-----An error happens.-----")
            self.theSession.message(anErrorMessage)
            self.theSession.message("---------------------------")

            # creates and display error message dialog.
            anErrorMessage = "An error happened! See MessageWindow."
            anErrorTitle = "An error happened!"
            if self['statusbar'] != None:
                self['statusbar'].push(1,anErrorMessage)
            anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
        else:

            self.__updatePropertyList()
            #self.thePluginManager.updateAllPluginWindow() 

    # end of updateValue


    def getSelectedFullPN( self ):
        anIter = self['theTreeView'].get_selection().get_selected()[1]
        if anIter == None:
			self.theSelectedFullPN = ''
        else:
            aSelectedProperty = self.theListStore.get_value( anIter,
                                                            PROPERTY_COL )
            self.theSelectedFullPN = convertFullIDToFullPN(
                                       self.theFullID(), aSelectedProperty )
        return self.theSelectedFullPN


    # ---------------------------------------------------------------
    # popupMenu
    #   - show popup menu
    #
    # aWidget         : widget
    # anEvent          : an event
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __popupMenu( self, aWidget, anEvent ):

        if anEvent.button == 3:  # 3 means right

            if self['theTreeView'].get_selection().get_selected()[1]==None :
                return None

            self.thePopupMenu.popup( None, None, None, 1, 0 )

    # end of poppuMenu

    def createNewPluginWindow( self, anObject ):

        # gets PluginWindowName from selected MenuItem
        aPluginWindowName = anObject.get_name()

        # gets selected property
        aRow = self['theTreeView'].get_selection().get_selected()[1]
        aSelectedProperty = self.theListStore.get_value(aRow,PROPERTY_COL)

        # creates RawFullPN
        aType = ENTITYTYPE_STRING_LIST[self.theFullID()[TYPE]] 
        anID = self.theFullID()[ID]
        aPath = self.theFullID()[SYSTEMPATH] 
        aRawFullPN = [(ENTITYTYPE_DICT[aType],aPath,anID,aSelectedProperty)]

        # creates PluginWindow
        self.thePluginManager.createInstance( aPluginWindowName, aRawFullPN )

    # end of createNewPluginWindow

    def updateViewAllProperties( self, *anObject ):
        isViewAll = self['checkViewAll'].get_active()
        if isViewAll:
            self.theDiscardList = []
            self.__updatePropertyList()
        else:
            self.theDiscardList = DISCARD_LIST 
            self.__updatePropertyList()


# ----------------------------------------------------------
# PropertyWindowPopupMenu -> gtk.Menu
#   - popup menu used by property window
# ----------------------------------------------------------
class PropertyWindowPopupMenu( gtk.Menu ):

    # ----------------------------------------------------------
    # Constructor
    #   - added PluginManager reference
    #   - added OsogoPluginWindow reference
    #   - acreates all menus
    #
    # aPluginManager : reference to PluginManager
    # aParent        : property window
    #
    # return -> None
    # This method is throwabe exception.
    # ----------------------------------------------------------
    def __init__( self, aPluginManager, aParent ):

        gtk.Menu.__init__(self)

        self.theParent = aParent
        self.thePluginManager = aPluginManager
        self.theMenuItem = {}

        # ------------------------------------------
        # initializes the size of menu
        # ------------------------------------------
        aMaxStringLength = 0
        aMenuSize = 0

        # ------------------------------------------
        # adds plugin window
        # ------------------------------------------
        for aPluginMap in self.thePluginManager.thePluginMap.keys():
            self.theMenuItem[aPluginMap]= gtk.MenuItem(aPluginMap)
            self.theMenuItem[aPluginMap].connect('activate', self.theParent.createNewPluginWindow )
            self.theMenuItem[aPluginMap].set_name(aPluginMap)
            self.append( self.theMenuItem[aPluginMap] )
            if aMaxStringLength < len(aPluginMap):
                aMaxStringLength = len(aPluginMap)
            aMenuSize += 1

        self.theWidth = (aMaxStringLength+1)*8
        #self.theHeight = (aMenuSize+1)*21 + 3
        self.theHeight = (aMenuSize+1)*21 + 3
        #self.set_usize( self.theWidth, self.theHeight )

        self.set_size_request( self.theWidth, self.theHeight )
        #self.append( gtk.MenuItem() )
        #self.set_size_request( 150, 450 )

    # end of __init__


    # ---------------------------------------------------------------
    # popup
    #    - shows this popup memu
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def popup(self, pms, pmi, func, button, time):

        # shows this popup memu
        gtk.Menu.popup(self, pms, pmi, func, button, time)
        self.show_all()

    # end of poup


# end of OsogoPluginWindowPopupMenu







if __name__ == "__main__":


    class simulator:

        dic={'PropertyList':
             ('PropertyList', 'ClassName', 'A','B','C','Substrate','Product'),
             'ClassName': ('MichaelisMentenProcess', ),
             'A': ('aaa', ) ,
             'B': (1.04E-3, ) ,
             'C': (41, ),
             'Substrate': ('Variable:/CELL/CYTOPLASM:ATP',
                           'Variable:/CELL/CYTOPLASM:ADP',
                           'Variable:/CELL/CYTOPLASM:AMP',
                           ),
             'Product': ('Variable:/CELL/CYTOPLASM:GLU',
                         'Variable:/CELL/CYTOPLASM:LAC',
                         'Variable:/CELL/CYTOPLASM:PYR',
                         ),
             'PropertyAttributes' : ('1','2','3','4','5','6','7','8'),
             } 

        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn[PROPERTY]]
    
    fpn = FullPropertyName('Process:/CELL/CYTOPLASM:MichaMen:PropertyName')



    def mainQuit( obj, data ):
        gtk.main_quit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.main()

    def main():
        aPluginManager = Plugin.PluginManager()
        aPropertyWindow = PropertyWindow( 'plugins', simulator(), [fpn,] ,aPluginManager)
        aPropertyWindow.addHandler( 'gtk_main_quit', mainQuit )
        aPropertyWindow.update()

        mainLoop()

    main()




