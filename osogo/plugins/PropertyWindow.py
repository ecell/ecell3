#!/usr/bin/env python

import string

#from PluginWindow import *
from OsogoPluginWindow import *
from ecell.ecssupport import *

#class PropertyWindow(PluginWindow):
class PropertyWindow(OsogoPluginWindow):

	def __init__( self, dirname, data, pluginmanager, root = None ):

		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
		self.thePluginManager.appendInstance( self ) 
		#self.initialize()

		# ------------------------------------------------------------------s
		self.addHandlers( { 'input_row_pressed'   : self.selectProperty,
		                    # 'show_button_pressed' : self.show,
		                     'window_exit'	  : self.exit } )
        
		self.thePropertyClist = self.getWidget( "property_clist" )
		self.theTypeEntry     = self.getWidget( "entry_TYPE" )
		self.theIDEntry       = self.getWidget( "entry_ID" )
		self.thePathEntry     = self.getWidget( "entry_PATH" )
		self.theClassNameEntry     = self.getWidget( "entry_NAME" )
		self.prevFullID = None

		if self.theRawFullPNList == ():
			return
		self.setFullPNList()
		# ------------------------------------------------------------------e

		if ( len( self.theFullPNList() ) > 1 ) and ( root != 'top_vbox' ):
			i = 1
			preFullID = self.theFullID()
			aClassName = self.__class__.__name__

		if root != 'top_vbox':
			if len( self.theFullPNList() ) > 1:
				self.addPopupMenu(1,1,1)
			else:
				self.addPopupMenu(0,1,1)
       
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

		self.theSelected = ''
        
		aNameFullPN = convertFullIDToFullPN( self.theFullID() ,'Name' )

		aNameList = list( self.theSession.theSimulator.getProperty( createFullPNString( aNameFullPN ) ) )

		aClassName = aNameList[0]
		self.theType =ENTITYTYPE_STRING_LIST[ self.theFullID()[TYPE] ]
		self.theID   = str( self.theFullID()[ID] )
		self.thePath = str( self.theFullID()[SYSTEMPATH] )
		aClassNameFullPN = convertFullIDToFullPN( self.theFullID(), 'ClassName' )
		aNameFullPN = convertFullIDToFullPN( self.theFullID(), 'Name' )

		aClassName = self.theSession.theSimulator.getProperty( createFullPNString( aClassNameFullPN ) )
		aName = self.theSession.theSimulator.getProperty( createFullPNString( aNameFullPN ) )


		self.theTypeEntry.set_text( self.theType + ' : ' + aClassName )
		self.theIDEntry.set_text  ( self.theID )
		self.thePathEntry.set_text( self.thePath )
		self.theClassNameEntry.set_text( aName )

		self.update()


	def update( self ):

		if self.prevFullID == self.theFullID():
			self.updatePropertyList()
			row = 0
			for aValue in self.theList:
				self.thePropertyClist.set_text(row,2,aValue[2])
				row += 1
		else:
			self.updatePropertyList()
			self.thePropertyClist.clear()
			for aValue in self.theList:
				self.thePropertyClist.append( aValue )


	
	def updatePropertyList( self ):

		self.theList = []

		aPropertyListFullPN = convertFullIDToFullPN( self.theFullID(),
		                                             'PropertyList' )

		self.prevFullID = convertFullPNToFullID( aPropertyListFullPN )        
		aPropertyList = self.theSession.theSimulator.getProperty( createFullPNString( aPropertyListFullPN ) )

		for aProperty in aPropertyList: # for (1)
			Set = -1
			aGet = -1

			if type(aProperty) == type(()):  # if (2)
				aSet = aProperty[1]
				aGet = aProperty[2]
				aProperty = aProperty[0]

			# end of if (2)

			if(aGet == 0):
				continue

			if (aProperty == 'ClassName'):

				#aFullPN = convertFullIDToFullPN( self.theFullID(), aProperty )
				#aValueList = self.theSession.theSimulator.getProperty( createFullPNString( aFullPN ) )

				#print aValueList
				pass

			elif (aProperty == 'PropertyList'):
				pass
			elif (aProperty == 'PropertyAttributes'):
				pass
			elif (aProperty == 'FullID'):
				pass
			elif (aProperty == 'ID'):
				pass
			elif (aProperty == 'Name'):
				pass
			else :
                
				aFullPN = convertFullIDToFullPN( self.theFullID(), aProperty )

				set = self.decodeAttribute( aSet )
				get = self.decodeAttribute( aGet )
                
				aFullPNString =  createFullPNString( aFullPN ) 

				aValueList = self.theSession.theSimulator.getProperty( createFullPNString( aFullPN ) )

				aDisplayedFlag = 0
				if type(aValueList) == type(()):
					if len(aValueList)  > 1 :
						aNumber = 1
						for aValue in aValueList :
							if type(aValue) == type(()):
								aValue = aValue[0]
							aList = [ aProperty, aNumber, aValue , get, set ]
							aList = map( str, aList )
							self.theList.append( aList ) 
							aNumber += 1
						aDisplayedFlag = 1

				if aDisplayedFlag == 0:
					aList = [ aProperty, '', aValueList , get, set]
					aList = map( str, aList )
					self.theList.append( aList )

			# end of if (2)

		# end of (1)


	def decodeAttribute(self, aAttribute):

		#data = {1 : ('-','+'),2 : ('+','-'),3 : ('+','+')}
		data = {0 : ('-'), 1 : ('+')}
		return data[ aAttribute ]


	def selectProperty(self, obj, data1, data2, data3):

		aSelectedItem = self.theList[data1]
		aFullPN = None

		try:
			aFullPropertyName = createFullPN( aSelectedItem[2] )
		except ValueError:
			pass

		if not aFullPropertyName:
			try:
				aFullID = createFullID( aSelectedItem[2] )
				aFullPN = convertFullIDToFullPN( aFullID )
			except ValueError:
				pass
            
		if not aFullPN:
			aFullPN = [ self.theType, self.thePath,
			            self.theID, aSelectedItem[0] ]

		self.theSelected = aFullPN


	#def show( self, obj ):
	#	print self.theSelected

   
if __name__ == "__main__":


    class simulator:

        dic={'PropertyList':
             ('PropertyList', 'ClassName', 'A','B','C','Substrate','Product'),
             'ClassName': ('MichaelisMentenReactor', ),
             'A': ('aaa', ) ,
             'B': (1.04E-3, ) ,
             'C': (41, ),
             'Substrate': ('Substance:/CELL/CYTOPLASM:ATP',
                           'Substance:/CELL/CYTOPLASM:ADP',
                           'Substance:/CELL/CYTOPLASM:AMP',
                           ),
             'Product': ('Substance:/CELL/CYTOPLASM:GLU',
                         'Substance:/CELL/CYTOPLASM:LAC',
                         'Substance:/CELL/CYTOPLASM:PYR',
                         ),
             'PropertyAttributes' : ('1','2','3','4','5','6','7','8'),
             } 

        def getProperty( self, fpn ):
            return simulator.dic[fpn[PROPERTY]]
    
    fpn = FullPropertyName('Reactor:/CELL/CYTOPLASM:MichaMen:PropertyName')



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









