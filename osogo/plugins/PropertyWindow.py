#!/usr/bin/env python

#from string import *
from OsogoPluginWindow import *

# column index of clist
PROPERTY_COL  = 0
VALUE_COL     = 1
GETABLE_COL   = 2
SETTABLE_COL  = 3

import gobject
import gtk
import os

PROPERTY_COL_TYPE=gobject.TYPE_STRING
VALUE_COL_TYPE=gobject.TYPE_STRING
GETABLE_COL_TYPE=gobject.TYPE_STRING
SETTABLE_COL_TYPE=gobject.TYPE_STRING


class PropertyWindow(OsogoPluginWindow):

	# ---------------------------------------------------------------
	# constructor
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):

		# calls superclass's constructor
		OsogoPluginWindow.__init__( self, aDirName, aData, aPluginManager, aRoot )
		self.theStatusBarWidget = None

	# end of __init__

	def openWindow( self ):
		#self.openWindow()
		OsogoPluginWindow.openWindow(self)
		# sets handlers
		self.addHandlers( { \
		      # property tree
		      'cursor_changed'                        : self.__selectProperty,
		      'on_property_clist_button_press_event'  : self.__popupMenu,
		      # button
		      'on_update_button_clicked'              : self.__updateValue })
		      #'window_exit'	                        : self.exit } )
        
		#self['property_clist'].connect('button_press_event',self.__popupMenu)

		# initializes buffer
		self.thePreFullID = None
		self.thePrePropertyMap = {}

		# initializes Iter
		self.theSelectedIter = None

		# initializes ListStore
		self.thePropertyListStore=gtk.ListStore(PROPERTY_COL_TYPE,
					    VALUE_COL_TYPE,
					    GETABLE_COL_TYPE,
					    SETTABLE_COL_TYPE)
		self['property_clist'].set_model(self.thePropertyListStore)
		renderer=gtk.CellRendererText()
		column=gtk.TreeViewColumn("Property",renderer,text=PROPERTY_COL)
		column.set_resizable(gtk.TRUE)
		self['property_clist'].append_column(column)
		column=gtk.TreeViewColumn("Value",renderer,text=VALUE_COL)
		column.set_resizable(gtk.TRUE)
		self['property_clist'].append_column(column)
		column=gtk.TreeViewColumn("Get",renderer,text=GETABLE_COL)
		column.set_resizable(gtk.TRUE)
		self['property_clist'].append_column(column)
		column=gtk.TreeViewColumn("Set",renderer,text=SETTABLE_COL)
		column.set_resizable(gtk.TRUE)
		self['property_clist'].append_column(column)
		
		# creates popu menu
		self.thePopupMenu =  PropertyWindowPopupMenu( self.thePluginManager, self )

		# initializes statusbar
		self.theStatusBarWidget = self['statusbar']

		if self.theRawFullPNList == ():
			return
                self.setIconList(
			os.environ['OSOGOPATH'] + os.sep + "ecell.png",
			os.environ['OSOGOPATH'] + os.sep + "ecell32.png")
		#self.__setFullPNList()
		self.update(True)

		#if ( len( self.theFullPNList() ) > 1 ) and ( aRoot != 'top_vbox' ):
		if ( len( self.theFullPNList() ) > 1 ) and ( aRoot != 'EntityWindow' ):
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
		#	anEntityStub = EntityStub( self.theSession.theSimulator, \
		#	               createFullIDString(self.theFullID()) )
		#
		#	for aProperty in anEntityStub.getPropertyList():
				# When a value is changed, 
		#		if self.thePrePropertyMap[aProperty] != anEntityStub.getProperty(aProperty):
		#			aChangedFlag = TRUE
		#			break

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
			self['entry_type'].set_text( ENTITYTYPE_STRING_LIST[self.theFullID()[TYPE]] )
			self['entry_classname'].set_text( anEntityStub.getClassname() )
			self['entry_id'].set_text( self.theFullID()[ID] )
			self['entry_path'].set_text( str( self.theFullID()[SYSTEMPATH] ) )

			# saves properties to buffer
			self.thePrePropertyMap = {}
			for aProperty in anEntityStub.getPropertyList():
				self.thePrePropertyMap[str(aProperty)] = [None, None]
				self.thePrePropertyMap[str(aProperty)][0] = anEntityStub.getProperty(aProperty)
				self.thePrePropertyMap[str(aProperty)][1] = anEntityStub.getPropertyAttributes(aProperty)
				
			# updates PropertyListStore
			self.__updatePropertyList()

		# save current full id to previous full id.
		self.preFullID = self.theFullID()

		# updates status bar
		if self['statusbar'] != None:
			self['statusbar'].push(1,'')
	
	# ---------------------------------------------------------------
	# __updatePropertyList
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __updatePropertyList( self ):

		self.theList = []
		#anEntityStub = EntityStub( self.theSession.theSimulator, createFullIDString(self.theFullID()) )
		aPropertyList = self.thePrePropertyMap.keys()

		# do nothing for following properties
		try:
			aPropertyList.remove( 'FullID' )
			aPropertyList.remove( 'Name' )
		except:
			pass

		for aPropertyName in aPropertyList: # for (1)

			aProperty = self.thePrePropertyMap[aPropertyName]
			anAttribute = aProperty[1]

			# When the getable attribute is false, value is ''
			if anAttribute[GETABLE] == FALSE:
				aValue = ''
			else:
				aValue = str( aProperty[0] )

			aSetString = decodeAttribute( anAttribute[SETTABLE] )
			aGetString = decodeAttribute( anAttribute[GETABLE] )

			aValueString = str( aValue )
			if( len( aValueString ) > 30 ):
				if type( aValue ) == list or\
				   type( aValue ) == tuple:
					aValueString = aValueString[:20]\
						       + ' .. (' +\
						       str( len( aValue ) ) +\
						       ' items)'
				else:
					aValueString = aValueString[:25]+ ' ..'

			aList = [ aPropertyName, aValueString, aGetString, aSetString ]
			self.theList.append( aList )

		self.thePropertyListStore.clear()

		for aValue in self.theList:
			iter=self.thePropertyListStore.append( )
			cntr=0
			for valueitem in aValue:
			    self.thePropertyListStore.set_value(iter,cntr,valueitem)
			    cntr+=1

	# end of __updatePropertyList


	# ---------------------------------------------------------------
	# updateValue
	#   - sets inputted value to the simulator
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __updateValue( self, anObject ):

		# ------------------------------------
		# gets inputted value from text field
		# ------------------------------------
		aValue = self['value_entry'].get_text()

		# ------------------------------------
		# gets selected number
		# ------------------------------------
		self.theSelectedIter = self['property_clist'].get_selection().get_selected()[1]

		# If nothing is selected, displays a confirm window.
		if self.theSelectedIter == None:
			aMessage = 'Select a property.'
			if self['statusbar'] != None:
				self['statusbar'].push(1,'Can\'t update inputted value. Select a Property.')
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			return None

		# ------------------------------------
		# gets getable status
		# ------------------------------------
		aGetable = self.thePropertyListStore.get_value(self.theSelectedIter,GETABLE_COL)

		# ------------------------------------
		# checks the type of inputted value 
		# ------------------------------------
		if aGetable == decodeAttribute(TRUE):
			aPreValue = self.theSession.theSimulator.getEntityProperty( \
			                        createFullPNString(self.theSelectedFullPN) ) 

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
					anErrorMessage = string.join( traceback.format_exception( \
			    		sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
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
					self.theSession.message("-----An error happens.-----")
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
			self['property_clist'].get_selection().select_iter(self.theSelectedIter)
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
		self.__selectProperty(None)
		return self.theSelectedFullPN

	# ---------------------------------------------------------------
	# __selectProperty
	#   - 
	#
	# anObject       : a selected object
	# aSelectedRow   : a selected row number
	# anObject1      : a dammy object
	# anObject2      : a dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __selectProperty(self, anObject):

		self.theSelectedIter = self['property_clist'].get_selection().get_selected()[1]

		# When selected row number is none, do nothing
		if self.theSelectedIter == None:
			self.theSelectedFullPN = ''
			return None

		# ---------------------------
		# sets selected full pn
		# ---------------------------
		aSelectedProperty = self.thePropertyListStore.get_value(self.theSelectedIter,PROPERTY_COL)
		self.theSelectedFullPN = convertFullIDToFullPN( self.theFullID(), aSelectedProperty )

		# ---------------------------
		# sets value to value entry
		# ---------------------------
		anEntityStub = EntityStub( self.theSession.theSimulator, createFullIDString(self.theFullID()) )
		aValue = anEntityStub.getProperty( aSelectedProperty )
		self['value_entry'].set_text( str(aValue) )

		# ---------------------------
		# sets sensitive of value entry
		# ---------------------------
		self.theSelectedIter = self['property_clist'].get_selection().get_selected()[1]
		aSetable = self.thePropertyListStore.get_value(self.theSelectedIter,SETTABLE_COL)

		if aSetable == decodeAttribute(TRUE):
			self['value_entry'].set_sensitive(TRUE)
			self['update_button'].set_sensitive(TRUE)
		else:
			self['value_entry'].set_sensitive(FALSE)
			self['update_button'].set_sensitive(FALSE)


	# end of selectedProperty


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

			if self['property_clist'].get_selection().get_selected()[1]==None :
				return None

			self.thePopupMenu.popup( None, None, None, 1, 0 )

	# end of poppuMenu

	def createNewPluginWindow( self, anObject ):

		# gets PluginWindowName from selected MenuItem
		aPluginWindowName = anObject.get_name()

		# gets selected property
		aRow = self['property_clist'].get_selection().get_selected()[1]
		aSelectedProperty = self.thePropertyListStore.get_value(aRow,PROPERTY_COL)

		# creates RawFullPN
		aType = ENTITYTYPE_STRING_LIST[self.theFullID()[TYPE]] 
		anID = self.theFullID()[ID]
		aPath = self.theFullID()[SYSTEMPATH] 
		aRawFullPN = [(ENTITYTYPE_DICT[aType],aPath,anID,aSelectedProperty)]

		# creates PluginWindow
		self.thePluginManager.createInstance( aPluginWindowName, aRawFullPN )

	# end of createNewPluginWindow
   

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
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aPluginManager = Plugin.PluginManager()
        aPropertyWindow = PropertyWindow( 'plugins', simulator(), [fpn,] ,aPluginManager)
        aPropertyWindow.addHandler( 'gtk_main_quit', mainQuit )
        aPropertyWindow.update()

        mainLoop()

    main()




