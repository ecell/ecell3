#!/usr/bin/env python

import string

from OsogoPluginWindow import *
from ecell.ecssupport import *

class VariableWindow( OsogoPluginWindow ):

	def __init__( self, dirname,  data, pluginmanager, root=None ):
        
		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )

		self.thePluginManager.appendInstance( self )    
		self.openWindow()
		#self.initialize()
		# ------------------------------------------------------------
		# 0 : not fixed  1: fixed
		self.theFixFlag = 0
        
		self.addHandlers( {'button_toggled': self.fix_mode,
#		                   'qty_increase_pressed': self.increaseValue, 
#		                   'qty_decrease_pressed': self.decreaseValue,
				   'on_value_spinbutton_changed' : self.changeValue,
#		                   'concentration_increase_pressed': self.increaseConcentration,
#		                   'concentration_decrease_pressed': self.decreaseConcentration,
		                   'input_value': self.changeValue,
		                   'input_concentration': self.inputConcentration,
		                   'window_exit' : self.exit } )

		self.theSession = pluginmanager.theSession

		aFullIDString = createFullIDString( self.theFullID() )

		self.theStub = self.theSession.createEntityStub( aFullIDString )
		self["id_label"].set_text( self.theStub.getName() )

		self.update()
		# ------------------------------------------------------------


	def update( self ):

		self.theValue = self.theStub.getProperty( 'Value' )
		self.theConcValue = self.theStub.getProperty( 'Concentration' )
		self.theIsFixed = self.theStub.getProperty( 'Fixed' )

		self['value_spinbutton'].set_text( str( self.theValue ) )
		self['concentration_entry'].set_text( str( self.theConcValue ) )

	def fix_mode( self, a ) :
		self.theFixFlag = not self.theFixFlag
		self.theStub.setProperty( 'Fixed', self.theFixFlag )

		self.thePluginManager.updateAllPluginWindow()

	def changeValue( self, obj ):

		aText = obj.get_text()

		string.strip(aText)

		if aText == '':
			return None
		else:
			self.theValue = string.atof( aText )
			self.theStub.setProperty( 'Value', self.theValue )

			self.thePluginManager.updateAllPluginWindow()


	def inputConcentration( self, obj ):
        
		self.theConcValue = string.atof( obj.get_text() )
		self.theStub.setProperty( 'Concentration', self.theConcValue )

		self.thePluginManager.updateAllPluginWindow()

	def increaseValue( self, button_object ):

		if self.theStub.getProperty( 'Value' ):
			self.theValue *= 2.0
		else:
			self.theValue = 1.0

		self.theStub.setProperty( 'Value', self.theValue )

		self.thePluginManager.updateAllPluginWindow()

	def decreaseValue( self, button_object ):

		self.theValue *= 0.5
		self.setProperty( 'Value', self.theValue )

		self.thePluginManager.updateAllPluginWindow()


	def increaseConcentration( self, button_object ):

		self.theConcValue *= 2.0
		self.setProperty( 'Concentration', self.theConcValue )

		self.thePluginManager.updateAllPluginWindow()

	def decreaseConcentration( self, button_object ):

		self.theConcValue *= 0.5
		self.setProperty( 'Concentration', self.theConcValue )

		self.thePluginManager.updateAllPluginWindow()
    
	def mainLoop():
		# FIXME: should be a custom function
		gtk.mainloop()

if __name__ == "__main__":

    class simulator:        
        dic={('Variable','/CELL/CYTOPLASM','ATP','Value') : (1950,),
             ('Variable','/CELL/CYTOPLASM','ATP','Concentration') : (0.353,),}
        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]
        
        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value
            
            
    fpn = ('Variable','/CELL/CYTOPLASM','ATP','')


    def mainQuit( obj, data ):
        gtk.mainquit()
         
    def mainLoop():
        # FIXME: should be a custom function
        
        gtk.mainloop()

    def main():
        aVariableWindow = VariableWindow( 'plugins', simulator(), [fpn,])
        aVariableWindow.addHandler( 'gtk_main_quit' , mainQuit )
        aVariableWindow.update()

        mainLoop()

    main()
    


