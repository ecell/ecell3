#!/usr/bin/env python


from OsogoPluginWindow import *
from ecell.ecssupport import *
import operator

# ------------------------------------------------------
# DigitalWindow -> OsogoPluginWindow
#   - show one numerical property 
# ------------------------------------------------------
class DigitalWindow( OsogoPluginWindow ):

	# ------------------------------------------------------
	# Constructor
	# 
	# aDirName:  directory name that includes glade file
	# data:      RawFullPN
	# aPluginManager
	# return -> None
	# ------------------------------------------------------
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):


		# calla constructor of superclass
		OsogoPluginWindow.__init__( self, aDirName, aData, aPluginManager, aRoot )
        
		#if type() self.theFullPN() 

		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )


		if operator.isNumberType( aValue ):
			#if operator.isNumberType( aValue[0] ):

			self.openWindow()
			self.thePluginManager.appendInstance( self )

			# ----------------------------------------------------------------
		#	self['toolbar5'].set_style( GTK.TOOLBAR_ICONS )
			self['increase_button'].set_relief( gtk.RELIEF_HALF )
			self['decrease_button'].set_relief( gtk.RELIEF_HALF )

			self.addHandlers( { 'input_value'    :self.inputValue,
		  	            'increase_value' :self.increaseValue,
			            'decrease_value' :self.decreaseValue,
			            'window_exit'    :self.exit,
			            'test'           :self.test } )

			aString = str( self.theFullPN()[ID] )
			aString += ':\n' + str( self.theFullPN()[PROPERTY] )
			self["id_label"].set_text( aString )
			self.update()
			# ----------------------------------------------------------------

		else:
			aMessage = "Error: (%s) is not numerical data" %aFullPNString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(0,aMessage,'Error!')


	def changeFullPN( self, anObject ):

		OsogoPluginWindow.changeFullPN( self, anObject )

		aString = str( self.theFullPN()[ID] )
		aString += ':\n' + str( self.theFullPN()[PROPERTY] )
		self["id_label"].set_text( aString )

	# end of changeFullPN


	def update( self ):

		#self["value_frame"].set_text( str( self.getValue( self.theFullPN() ) ) )
		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		self["value_frame"].set_text( str( aValue ) )


	def inputValue( self, obj ):
		aValue =  string.atof(obj.get_text())
		self.setValue( self.theFullPN(), aValue )


	def increaseValue( self, obj ):

		if self.getValue( self.theFullPN() ):
			self.setValue( self.theFullPN(), self.getValue( self.theFullPN() ) * 2.0 )
		else:
			self.setValue( self.theFullPN(), 1.0 )
        

	def decreaseValue( self, obj ):

		self.setValue( self.theFullPN(), self.getValue( self.theFullPN() ) * 0.5 )

			
	def test( self, obj ):
		print 'you did it'

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









