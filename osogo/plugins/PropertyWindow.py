#!/usr/bin/env python

#from string import *
from OsogoPluginWindow import *

# column index of clist
PROPERTY_COL  = 0
NUMBER_COL    = 1
VALUE_COL     = 2
GETABLE_COL   = 3
SETTABLE_COL  = 4
import gobject
import gtk
PROPERTY_COL_TYPE=gobject.TYPE_STRING
NUMBER_COL_TYPE=gobject.TYPE_STRING
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


		OsogoPluginWindow.__init__( self, aDirName, aData, aPluginManager, aRoot )
		self.openWindow()
		print self.getWidget("PropertyWindow")
		print self.widgets
		
		self.thePluginManager.appendInstance( self ) 

		self.addHandlers( { 'cursor_changed'             : self.selectProperty,
		                    'on_property_clist_button_press_event'  : self.popupMenu,
		                    'on_update_button_pressed'              : self.updateValue,
		                    'window_exit'	                        : self.exit } )
        
		self.thePropertyCList     = self.getWidget( "property_clist" )
		self.theTypeEntry         = self.getWidget( "entry_TYPE" )
		self.theIDEntry           = self.getWidget( "entry_ID" )
		self.thePathEntry         = self.getWidget( "entry_PATH" )
		self.theClassNameEntry    = self.getWidget( "entry_NAME" )
		self.preFullID = None

		self.theSelectedFullPNList = []
		self.theSelectedRowNumber = -1
		self.thePropertyListStore=gtk.ListStore(PROPERTY_COL_TYPE,
					    NUMBER_COL_TYPE,
					    VALUE_COL_TYPE,
					    GETABLE_COL_TYPE,
					    SETTABLE_COL_TYPE)
		self.thePropertyCList.set_model(self.thePropertyListStore)
		renderer=gtk.CellRendererText()
		column=gtk.TreeViewColumn("Property",renderer,text=PROPERTY_COL)
		self.thePropertyCList.append_column(column)
		column=gtk.TreeViewColumn("Number",renderer,text=NUMBER_COL)
		self.thePropertyCList.append_column(column)
		column=gtk.TreeViewColumn("Value",renderer,text=VALUE_COL)
		self.thePropertyCList.append_column(column)
		column=gtk.TreeViewColumn("Gettable",renderer,text=GETABLE_COL)
		self.thePropertyCList.append_column(column)
		column=gtk.TreeViewColumn("Settable",renderer,text=SETTABLE_COL)
		self.thePropertyCList.append_column(column)
		
		
					
		self.thePopupMenu =  PropertyWindowPopupMenu( self.thePluginManager, self )

		if self.theRawFullPNList == ():
			return
		self.setFullPNList()

		if ( len( self.theFullPNList() ) > 1 ) and ( aRoot != 'top_vbox' ):
			self.preFullID = self.theFullID()
			aClassName = self.__class__.__name__
	
	# end of __init__

       
	# ---------------------------------------------------------------
	# setFullPNList
	#  - this method is called from EntryList Window, when users select 
	#    some entries.
	#  - display all properties of selected entity.
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setFullPNList( self ):

		#self.theSelected = ''
		self.theSelectedFullPN = ''
        
		aNameFullPN = convertFullIDToFullPN( self.theFullID() ,'Name' )

		aClassName = self.theSession.theSimulator.getEntityProperty( createFullPNString( aNameFullPN ) ) 

		self.theType =ENTITYTYPE_STRING_LIST[ self.theFullID()[TYPE] ]
		self.theID   = str( self.theFullID()[ID] )
		self.thePath = str( self.theFullID()[SYSTEMPATH] )

		aClassName = self.theSession.theSimulator.getEntityClassName( createFullIDString( self.theFullID() ) )
		aName = self.theSession.theSimulator.getEntityProperty( createFullPNString( aNameFullPN ) )

		self.theTypeEntry.set_text( self.theType + ' : ' + aClassName )
		self.theIDEntry.set_text( self.theID )
		self.thePathEntry.set_text( self.thePath )
		self.theClassNameEntry.set_text( aName )

		self.update()

	# end of setFullPNList


	# ---------------------------------------------------------------
	# update (overwrite the method of superclass)
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def update( self ):

		self.updatePropertyList()

		# if current full id matches previous full id,
		# then does nothing.
		if self.preFullID == self.theFullID:
			pass

		# if current full id doesn't match previous full id,
		# then rewrite all property of clist.
		else:
			self.thePropertyListStore.clear()
			for aValue in self.theList:
				iter=self.thePropertyListStore.append( )
				cntr=0
				for valueitem in aValue:
				    self.thePropertyListStore.set_value(iter,cntr,valueitem)
				    cntr+=1

		# save current full id to previous full id.
		self.preFullID = self.theFullID()

		self['value_entry'].set_text('')
		self['value_entry'].set_sensitive(FALSE)
		self['update_button'].set_sensitive(FALSE)
	
	# ---------------------------------------------------------------
	# updatePropertyList
	#   - 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updatePropertyList( self ):

		self.theList = []

		aPropertyList = self.theSession.theSimulator.getEntityPropertyList( createFullIDString( self.theFullID() ) )

		for aProperty in aPropertyList: # for (1)

			# does nothing for following property
			if aProperty=='PropertyList' or \
			   aProperty=='ClassName' or \
			   aProperty=='PropertyAttributes' or \
			   aProperty=='FullID' or \
			   aProperty=='ID' or \
			   aProperty=='Name':

				# does nothing.
				continue

			aFullPN = convertFullIDToFullPN( self.theFullID(), aProperty )
			anAttribute = self.theSession.theSimulator.getEntityPropertyAttributes( \
			                                              createFullPNString( aFullPN ) )

			# When the getable attribute is false, value is ''
			if anAttribute[GETABLE] == FALSE:
				aValueList = ''
			else:
				aValueList = self.theSession.theSimulator.getEntityProperty( createFullPNString( aFullPN ) )

			aSetString = decodeAttribute( anAttribute[SETTABLE] )
			aGetString = decodeAttribute( anAttribute[GETABLE] )
                
			aFullPNString =  createFullPNString( aFullPN ) 


			aDisplayedFlag = 0
			if type(aValueList) == type(()):
				if len(aValueList)  > 1 :
					aNumber = 1
					for aValue in aValueList :
						#if type(aValue) == type(()):
						#	aValue = aValue[0]
						aList = [aProperty, aNumber, aValue , aGetString, aSetString ]
						aList = map( str, aList )
						self.theList.append( aList ) 
						aNumber += 1
					aDisplayedFlag = 1

			if aDisplayedFlag == 0:
				aList = [ aProperty, '', aValueList , aGetString, aSetString ]
				aList = map( str, aList )
				self.theList.append( aList )

		row = 0
#		for aValue in self.theList:
#			self.thePropertyCList.set_text(row,2,aValue[2])
#			row += 1
		for aValue in self.theList:
			iter=self.thePropertyListStore.append( )
			cntr=0
			for valueitem in aValue:
			    self.thePropertyListStore.set_value(iter,cntr,valueitem)
			    cntr+=1

	# end of updatePropertyList

	# ---------------------------------------------------------------
	# updateValue
	#   - sets inputted value to the simulator
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updateValue( self, anObject ):

		# ------------------------------------
		# gets inputted value from text field
		# ------------------------------------
		aValue = self['value_entry'].get_text()

		# ------------------------------------
		# gets selected number
		# ------------------------------------
		self.theSelectedRowNumber = self.thePropertyCList.get_selection().get_selected()[1]

		# ------------------------------------
		# gets getable status
		# ------------------------------------
		aGetable = self.thePropertyListStore.get_value(self.theSelectedRowNumber,GETABLE_COL)

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
					anErrorMessage = "The inputted value must be integer!"
					anErrorTitle = "The type error!"
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
					anErrorMessage = "The inputted value must be float!"
					anErrorTitle = "The type error!"
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
					anErrorMessage = "The inputted value must be tuple!"
					anErrorTitle = "The type error!"
					anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
					return None


		# ---------------------------------------------------
		# when the number column is not blank, reate tuple.
		# ---------------------------------------------------
		aPropertyValue = []
		aNumber = self.thePropertyListStore.get_value(self.theSelectedRowNumber,NUMBER_COL)
		if aNumber != '':

			aSelectedProperty = self.thePropertyListStore.get_value(self.theSelectedRowNumber,PROPERTY_COL)
			#print "aSelectedProperty = %s" %aSelectedProperty
			aRow=self.thePropertyListStore.get_iter_first()
			while aRow!=None:

				if aSelectedProperty != self.thePropertyListStore.get_value(aRow,PROPERTY_COL):
					continue

				if aRow == self.theSelectedRowNumber:
					aPropertyValue.append( aValue )
				else:
					aCListValue = self.thePropertyListStore.get_value(aRow,VALUE_COL) 
					aCListValue = convertStringToTuple( aCListValue )
					aPropertyValue.append( aCListValue )

			aPropertyValue = tuple(aPropertyValue)

		else:
			aPropertyValue = aValue


		aFullPNString = createFullPNString(self.theSelectedFullPN)

		#if aGetable == decodeAttribute(TRUE):
		#	print self.theSession.theSimulator.getEntityProperty( createFullPNString(self.theSelectedFullPN) ) 
		try:
			#self.theSession.theSimulator.setEntityProperty( createFullPNString(self.theSelectedFullPN), aPropertyValue ) 
			#self.setValue( createFullPNString(self.theSelectedFullPN), aPropertyValue ) 
			#print self.theSelectedFullPN
			self.setValue( self.theSelectedFullPN, aPropertyValue ) 
		except:

			# print out traceback
			import sys
			import traceback
			anErrorMessage = string.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.theSession.message("-----An error happens.-----")
			self.theSession.message(anErrorMessage)
			self.theSession.message("---------------------------")

			# creates and display error message dialog.
			anErrorMessage = "An error happened!\nSee MessageWindow."
			anErrorTitle = "An error happened!"
			anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)

		else:

			self.updatePropertyList()
			#self.thePluginManager.updateAllPluginWindow() 

	# end of updateValue



	# ---------------------------------------------------------------
	# selectProperty
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
	def selectProperty(self, anObject):
		self.theSelectedRowNumber = self.thePropertyCList.get_selection().get_selected()[1]
		self.theSelectedFullPN = ''

		# ---------------------------
		# sets selected full pn
		# ---------------------------
		aProperty = self.thePropertyListStore.get_value(self.theSelectedRowNumber,PROPERTY_COL)
		aType = string.strip( string.split(self.theTypeEntry.get_text(),':')[0] )
		anID = self.theIDEntry.get_text()
		aPath = self.thePathEntry.get_text()

		self.theSelectedFullPN = (ENTITYTYPE_DICT[aType],aPath,anID,aProperty)

		# ---------------------------
		# sets value
		# ---------------------------
		aValue = self.thePropertyListStore.get_value(self.theSelectedRowNumber,VALUE_COL)
		self['value_entry'].set_text(aValue)

		# ---------------------------
		# sets sensitive
		# ---------------------------
		self.theSelectedRowNumber = self.thePropertyCList.get_selection().get_selected()[1]
		aSetable = self.thePropertyListStore.get_value(self.theSelectedRowNumber,SETTABLE_COL)

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
	def popupMenu( self, aWidget, anEvent ):

		if anEvent.button == 3:  # 3 means right

			if self['property_clist'].get_selection().get_selected()[1]==None :
				return None

			self.theSelectedRowNumber = self['property_clist'].get_selection().get_selected()[1]
			aGetable = self.thePropertyListStore.get_value(self.theSelectedRowNumber,GETABLE_COL)

			if aGetable == decodeAttribute(TRUE):
				self.thePopupMenu.popup( None, None, None, 1, 0 )

	# end of poppuMenu

	def createNewPluginWindow( self, anObject ):

		aPluginWindowName = anObject.get_name()
		aRow = self.thePropertyCList.get_selection().get_selected()[1]
		aProperty = self.thePropertyListStore.get_value(aRow,PROPERTY_COL)
		aType = string.strip( string.split(self.theTypeEntry.get_text(),':')[0] )
		anID = self.theIDEntry.get_text()
		aPath = self.thePathEntry.get_text()

		aRawFullPN = [(ENTITYTYPE_DICT[aType],aPath,anID,aProperty)]
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
		# adds plugin window
		# ------------------------------------------
		for aPluginMap in self.thePluginManager.thePluginMap.keys():
			self.theMenuItem[aPluginMap]= gtk.MenuItem(aPluginMap)
			self.theMenuItem[aPluginMap].connect('activate', self.theParent.createNewPluginWindow )
			self.theMenuItem[aPluginMap].set_name(aPluginMap)
			self.append( self.theMenuItem[aPluginMap] )

		#self.append( gtk.MenuItem() )


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




