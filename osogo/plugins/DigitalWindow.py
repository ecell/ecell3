#!/usr/bin/env python


from OsogoPluginWindow import *
from ecell.ecssupport import *
import operator
import ConfirmWindow

# ------------------------------------------------------
# DigitalWindow -> OsogoPluginWindow
#   - show one numerical property 
# ------------------------------------------------------
class DigitalWindow( OsogoPluginWindow ):

	# ------------------------------------------------------
	# Constructor
	# 
	# aDirName(str)   : directory name that includes glade file
	# data            : RawFullPN
	# aPluginManager  : the reference to pluginmanager 
	# return -> None
	# ------------------------------------------------------
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):

		# call constructor of superclass
		OsogoPluginWindow.__init__( self, aDirName, aData, aPluginManager, aRoot )
        
		#if type() self.theFullPN() 

		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )

		if operator.isNumberType( aValue ):
			self.openWindow()
			self.thePluginManager.appendInstance( self )

			self.addHandlers( { 
			            'on_value_frame_changed' :self.inputValue,
		  	            'on_increase_button_clicked' :self.increaseValue,
		  	            'on_decrease_button_clicked' :self.decreaseValue,
			            'on_DigitalWindow_delete_event'    :self.exit } )

			aString = str( self.theFullPN()[ID] )
			aString += ':\n' + str( self.theFullPN()[PROPERTY] )
			self["id_label"].set_text( aString )
			self.update()
			# ----------------------------------------------------------------

		else:
			aMessage = "Error: (%s) is not numerical data" %aFullPNString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(0,aMessage,'Error!')

	# end of __init__


	# ------------------------------------------------------
	# changeFullPN
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def changeFullPN( self, anObject ):

		OsogoPluginWindow.changeFullPN( self, anObject )

		aString = str( self.theFullPN()[ID] )
		aString += ':\n' + str( self.theFullPN()[PROPERTY] )
		self["id_label"].set_text( aString )

	# end of changeFullPN


	# ------------------------------------------------------
	# update
	# 
	# return -> None
	# ------------------------------------------------------
	def update( self ):

		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		self["value_frame"].set_text( str( aValue ) )

	# end of update


	# ------------------------------------------------------
	# inputValue
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def inputValue( self, obj ):

		# gets text from text field.
		aText = string.split(self['value_frame'].get_text())
		if type(aText) == type([]):
			if len(aText) > 0:
				aText = aText[0]
			else:
				return None
		else:
			return None

		# Only when the length of text > 0,
		# checks type of text and set it.
		if len(aText)>0:
			# Only when the value is numeric, 
			# the value will be set to value_frame.
			try:
				aValue = string.atof( aText )
				self.setValue( self.theFullPN(), aValue )
			except:
				ConfirmWindow.ConfirmWindow(0,'Input numerical value.')
				aValue = self.getValue( self.theFullPN() )
				self["value_frame"].set_text( str( aValue ) )
			return None
		else:
			return None

	# end of inputValue


	# ------------------------------------------------------
	# increaseValue
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def increaseValue( self, obj ):

		if self.getValue( self.theFullPN() ):
			self.setValue( self.theFullPN(), self.getValue( self.theFullPN() ) * 2.0 )
		else:
			self.setValue( self.theFullPN(), 1.0 )

	# end of increaseValue
		
        
	# ------------------------------------------------------
	# decreaseValue
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def decreaseValue( self, obj ):

		self.setValue( self.theFullPN(), self.getValue( self.theFullPN() ) * 0.5 )

	# end of decreaseValue
			
# end of DigitalWindow


### test code

if __name__ == "__main__":

    class simulator:

        dic={('Variable', '/CELL/CYTOPLASM', 'ATP','Value') : (1950,),}

        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]

        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Variable','/CELL/CYTOPLASM','ATP','Value')

    def mainQuit( obj, aData ):
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aDigitalWindow = DigitalWindow( 'plugins', simulator(), [fpn,] )
        aDigitalWindow.addHandler( 'gtk_main_quit', mainQuit )
        aDigitalWindow.update()

        mainLoop()

    main()









