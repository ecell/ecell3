
from ecell.ecssupport import *

import gobject
import gtk
import gtk.gdk
import operator
from ConfirmWindow import *

VARREF_NAME = 0
VARREF_FULLID = 1
VARREF_COEF = 3

NAME_COLUMN = 0
FULLID_COLUMN = 1
COEF_COLUMN = 2
EDITABLE_COLUMN = 3
COLOR_COLUMN = 4


class VariableReferenceEditor:
    
    def __init__( self, aParent, attachmentPoint ):
        self.theParent = aParent
        self.theSession = self.theParent.theSession

        self.theListStore = gtk.ListStore( gobject.TYPE_STRING,
                                            gobject.TYPE_STRING,
                                            gobject.TYPE_STRING,
                                            gobject.TYPE_BOOLEAN )
        self.theTreeView = gtk.TreeView()
        self.theTreeView.show()
        self.theTreeView.set_model( self.theListStore )

        rendererName = gtk.CellRendererText( )
        rendererName.connect('edited', self.__cellEdited, NAME_COLUMN)
        columnName=gtk.TreeViewColumn( "Name", rendererName, text=NAME_COLUMN, editable = EDITABLE_COLUMN)
        columnName.set_visible( True )
        columnName.set_resizable( True )
        columnName.set_reorderable( True )
        columnName.set_sort_column_id = NAME_COLUMN
        
        rendererFullID = gtk.CellRendererText()
        rendererFullID.connect('edited', self.__cellEdited, FULLID_COLUMN)
        columnFullID =gtk.TreeViewColumn( "FullID", rendererFullID, text=FULLID_COLUMN, editable=EDITABLE_COLUMN)
        columnFullID.set_visible( True )
        columnFullID.set_resizable( True )
        columnFullID.set_reorderable( True )
        columnFullID.set_sort_column_id ( FULLID_COLUMN )
        
        rendererCoef = gtk.CellRendererText()
        rendererCoef.connect('edited', self.__cellEdited, COEF_COLUMN)
        columnCoef=gtk.TreeViewColumn( "Coefficient", rendererCoef, text=COEF_COLUMN, editable=EDITABLE_COLUMN)
        columnCoef.set_visible( True )
        columnCoef.set_resizable( True )
        columnCoef.set_reorderable( True )
        columnCoef.set_sort_column_id ( COEF_COLUMN )
        
        self.theTreeView.append_column( columnName )
        self.theTreeView.append_column( columnFullID )
        self.theTreeView.append_column( columnCoef )
        attachmentPoint.add( self.theTreeView )
        self.setDisplayedFullID ( self.theParent.theFullID() )
        
        
    def setDisplayedFullID ( self, aFullID ):
        self.theFullID = aFullID
        self.theFullIDString = createFullIDString( self.theFullID )
        self.theFullPNString = createFullIDString( self.theFullID ) + ":VariableReferenceList"
        self.update()
        
    def getProcessFullID( self ):
        return self.theFullID
        
    def update( self ):
        # gets varreflist
        theValue = self.theSession.theSimulator.getEntityProperty( self.theFullPNString )
        
        #redraw whole list
        self.theListStore.clear()
        anIter = None
        for aVariableReference in theValue:
            aName = aVariableReference[VARREF_NAME]
            aFullID = aVariableReference[VARREF_FULLID]
            aCoef = aVariableReference[VARREF_COEF]
            anIter = self.theListStore.append(  )
            self.theListStore.set( anIter, 0, aName, 1, aFullID, 2, aCoef, 3, gtk.TRUE )
        
        
    def __setValue( self ):
        #take value from list and redraw
        aVarrefList = []
        anIter = self.theListStore.get_iter_first()
        while anIter != None:
            aVarref = ["","",1, 1]
            aVarref[VARREF_NAME] = self.theListStore.get_value( anIter, NAME_COLUMN )
            aVarref[VARREF_FULLID] = self.theListStore.get_value( anIter, FULLID_COLUMN )
            aVarref[VARREF_COEF] = self.theListStore.get_value( anIter, COEF_COLUMN )
            aVarrefList.append( tuple( aVarref ) )
            anIter = self.theListStore.iter_next( anIter )
        aVarrefListTuple = tuple( aVarrefList )

        try:
            self.theSession.theSimulator.setEntityProperty( self.theFullPNString, aVarrefListTuple )
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
            anErrorMessage = "Couldnot change variablereference"
            anErrorTitle = "Error"
            anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
            self.update()
            
        
    def __cellEdited( self, *args ):
        aNewValue = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        column = args[3]
        if column == COEF_COLUMN:
            #check whether it is integer
            if not operator.isNumberType( aNewValue):
                anErrorWindow = ConfirmWindow(OK_MODE,"Coefficient should be numeric.","Error")
                return
        self.theListStore.set_value( anIter, column, aNewValue )
        self.__setValue()
        
    def __buttonPressed( self, *args ):
        event = args[2]
        if event.button == gtk.gdk._2BUTTON_PRESS:
            self.__popUpMenu()
            return gtk.TRUE
            
    def __popUpMenu(self ):
        selectedIter = self.__getSelectedIter()
        aMenu = gtk.Menu()
        selectedFullID = self.__getSelectedFullID()
        if selectedFullID == None:
            isFullIDReal = False
        else:
            isFullIDReal = self.__doesExistEntity( selectedFullID )
            
        openItem = gtk.MenuItem( "Open" )
        if isFullIDReal:
            openItem.connect( "activate", self.__openAction, selectedFullID )
        else:
            openItem.set_sensitive( gtk.FALSE )
        aMenu.append( openItem )

        openNewItem = gtk.MenuItem( "Open in new" )
        if isFullIDReal:
            openNewItem.connect( "activate", self.__openNewAction, selectedFullID )
        else:
            openNewItem.set_sensitive( gtk.FALSE )
        aMenu.append( openNewItem )
        aMenu.append( gtk.SeparatorMenuItem() )
        
        addItem = gtk.MenuItem( "Add" )
        addItem.connect("activate", self.__addAction )
        aMenu.append( addItem )
        
        deleteItem = gtk.MenuItem( "Delete" )

        if selectedIter != None:
            deleteItem.connect( "activate", self.__deleteAction, anIter )
        else:
            deleteItem.set_sensitive( gtk.FALSE )
        aMenu.append( addItem )
        aMenu.popup( None, None, None, 1, 0 )
    
    
    def __openAction( self, *args ):
        aFullIDString = args[1]
        self.theSession.thePluginManager.createInstance( "PropertyWindow", 
                        [convertFullIDToFullPN( aFullIDString.split(':') )] )
                        
    def __openNewAction ( self, *args ):
        pass
        
    def __addAction ( self, *args ):
        pass
        
    def __deleteAction( self, *args ):
        pass
                        
    


    def __getSelectedFullID( self ):
        anIter = self.__getSelectedIter()
        if anIter == None:
            return None
        aVarref = self.theListStore.get_value( anIter, FULLID_COLUMN )
        return  self.__getAbsoluteReference( aVarref )
        
    
    def __getSelectedIter( self ):
        anIter = self['theTreeView'].get_selection().get_selected()[1]
        return anIter
        

        
    def __doesExistEntity( self, anEntity ):
        try:
            self.theSession.theSimulator.getEntityClassName( anEntity )
        except:
            return False
        else:
            return True


    def __getAbsoluteReference( self, aVariableRef ):
        if __self.isAbsoluteReference( aVariableRef ):
            return aVariableRef
        aVariable = aVariableRef.split(':')
        if aVariable[1][0] == '/':
            # absolute ref
            absolutePath = aVariable[1]
        elif aVariable[1][0] == '.':
            aProcess = self.theFullIDString.split(':')[1]
            aProcessPath = aProcess.split('/')
            while True:
                if len(aProcessPath) == 0:
                    break
                if aProcessPath[0] == '':
                    aProcessPath.__delitem__(0)
                else:
                    break
            aVariablePath = aVariable[1].split('/')
            absolutePath = ''
            while aVariablePath != []:
                pathString =  aVariablePath.pop()
                if pathString == '.':
                    break
                elif pathString == '..':
                    if len(aProcessPath) == 0:
                        return aVariableRef
                    aProcessPath.pop()
                else:
                    absolutePath =  pathString + '/' + absolutePath
            oldPath = '/' + '/'.join(aProcessPath)
            absolutePath = absolutePath.rstrip('/')
            if oldPath != '/' and absolutePath != '':
                oldPath +='/'
            absolutePath =  oldPath + absolutePath
    
        else:
            return aVariableRef
    
        return "Variable" + ':' + absolutePath + ':' + aVariable[2]
    
    
    
    def __isAbsoluteReference( aVariableRef ):
        aList = aVariableRef.split(':')
        return aList[1][0] == '/'
    
