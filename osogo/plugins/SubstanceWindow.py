#!/usr/bin/env python

import string

from OsogoPluginWindow import *
from ecell.ecssupport import *
import GTK

class SubstanceWindow( OsogoPluginWindow ):

	def __init__( self, dirname,  data, pluginmanager, root=None ):
        
		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )

		self.thePluginManager.appendInstance( self )    
		#self.initialize()
		# ------------------------------------------------------------
		# 0 : not fixed  1: fixed
		self.theFixFlag = 0
        
		self['toolbar1'].set_style( GTK.TOOLBAR_ICONS )
		self['toolbar1'].set_button_relief( GTK.RELIEF_HALF )
		self['toolbar2'].set_style( GTK.TOOLBAR_ICONS )
		self['toolbar2'].set_button_relief( GTK.RELIEF_HALF )
		self['toolbar3'].set_style( GTK.TOOLBAR_ICONS )
		self['toolbar3'].set_button_relief( GTK.RELIEF_HALF )
		self['toolbar4'].set_style( GTK.TOOLBAR_ICONS )
		self['toolbar4'].set_button_relief( GTK.RELIEF_HALF )        
        
		self.addHandlers( {'button_toggled': self.fix_mode,
		                   'qty_increase_pressed': self.increaseQuantity, 
		                   'qty_decrease_pressed': self.decreaseQuantity,
		                   'concentration_increase_pressed': self.increaseConcentration,
		                   'concentration_decrease_pressed': self.decreaseConcentration,
		                   'input_quantity': self.inputQuantity,
		                   'input_concentration': self.inputConcentration,
		                   'window_exit' : self.exit } )

		self.theQtyFPN = convertFullIDToFullPN( self.theFullID(), 'Quantity' )
		self.theConcFPN = convertFullIDToFullPN( self.theFullID(), 'Concentration' )

		aFullIDString = createFullIDString( self.theFullID() )
		self["id_label"].set_text( aFullIDString )

		self.update()
		# ------------------------------------------------------------

		self.theSession = pluginmanager.theSession
        
		#if len( self.theFullPNList() ) > 1:
		#	i = 1
		#	preFullID = self.theFullID()
		#	aClassName = self.__class__.__name__
		#	while i < len( self.theFullPNList() ):
		#		aFullID = self.theFullIDList()[i]
		#		if aFullID != preFullID:
		#			self.thePluginManager.createInstance( aClassName, (self.theFullPNList()[i],), root)
		#		preFullID = aFullID
		#		i = i + 1

        
		if len( self.theFullPNList() ) > 1:
			self.addPopupMenu(1,1,1)
		else:
			self.addPopupMenu(0,1,1)


	def update( self ):

		self.theQtyValue = self.getValue( self.theQtyFPN )
		self.theConcValue = self.getValue( self.theConcFPN )
		self['Quantity_entry'].set_text( str( self.theQtyValue ) )
		self['Concentration_entry'].set_text( str( self.theConcValue ) )
    

	def fix_mode( self, a ) :
		self.theFixFlag = 1 - self.theFixFlag
		if self.theFixFlag == 0:
			self.theSession.printMessage( "not fixed\n" )
		else:
			self.theSession.printMessage( "fixed\n" )


	def inputQuantity( self, obj ):

		self.theQtyValue = string.atof( obj.get_text() )
		self.setValue( self.theQtyFPN, self.theQtyValue )


	def inputConcentration( self, obj ):
        
		self.theConcValue = string.atof( obj.get_text() )
		self.setValue( self.theConcFPN, self.theConcValue )


	def increaseQuantity( self, button_object ):

		if self.getValue( self.theQtyFPN ):
			self.theQtyValue *= 2.0
		else:
			self.theQtyValue = 1.0

		self.setValue( self.theQtyFPN, self.theQtyValue )


	def decreaseQuantity( self, button_object ):

		self.theQtyValue *= 0.5
		self.setValue( self.theQtyFPN, self.theQtyValue )


	def increaseConcentration( self, button_object ):

		self.theConcValue *= 2.0
		self.setValue( self.theConcFPN, self.theConcValue )


	def decreaseConcentration( self, button_object ):

		self.theConcValue *= 0.5
		self.setValue( self.theConcFPN, self.theConcValue )

    
	def mainLoop():
		# FIXME: should be a custom function
		gtk.mainloop()

if __name__ == "__main__":

    class simulator:        
        dic={('Substance','/CELL/CYTOPLASM','ATP','Quantity') : (1950,),
             ('Substance','/CELL/CYTOPLASM','ATP','Concentration') : (0.353,),}
        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]
        
        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value
            
            
    fpn = ('Substance','/CELL/CYTOPLASM','ATP','')


    def mainQuit( obj, data ):
        gtk.mainquit()
         
    def mainLoop():
        # FIXME: should be a custom function
        
        gtk.mainloop()

    def main():
        aSubstanceWindow = SubstanceWindow( 'plugins', simulator(), [fpn,])
        aSubstanceWindow.addHandler( 'gtk_main_quit' , mainQuit )
        aSubstanceWindow.update()

        mainLoop()

    main()
    


