#!/usr/bin/env python2

import string

from OsogoPluginWindow import *
from ecell.ecssupport import *

import Numeric
import GTK
import operator

class BargraphWindow( OsogoPluginWindow ):
    
	def __init__( self, dirname, data, pluginmanager, root=None ):

		OsogoPluginWindow.__init__( self, dirname, data, pluginmanager, root )
		self.theSession = pluginmanager.theSession
		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )

		#if operator.isNumberType( aValue[0] ):
		if operator.isNumberType( aValue ):
			self.openWindow()
			self.thePluginManager.appendInstance( self )   
			#self.initialize()
			# -------------------------------------------------
			self['toolbar5'].set_style( GTK.TOOLBAR_ICONS )
			self['toolbar6'].set_style( GTK.TOOLBAR_ICONS )
			self['toolbar5'].set_button_relief( GTK.RELIEF_HALF )
			self['toolbar6'].set_button_relief( GTK.RELIEF_HALF )        
        
			self.pull = FALSE
			self.thePositiveFlag = TRUE
			self.theFixFlag = FALSE
			self.theActualValue = FALSE
			self.theBarLength = FALSE
			self.theMultiplier = FALSE
        
			self.addHandlers( { \
		                   'on_add_button_clicked'      : self.updateByIncrease,
		                   'on_subtract_button_clicked' : self.updateByDecrease,
		                   'multiplier_entry_activate'  : self.updateByTextentry,
		                   'fix_checkbutton_toggled'    : self.updateByFix ,
		                   'window_exit'                : self.exit })
        
			self.theIDEntry = self.getWidget( "property_id_label" )
			self.theMultiplier1Entry = self.getWidget("multiplier1_label")
			self.update()

			# -------------------------------------------------

		else:
			aMessage = "Error: (%s) is not numerical data" %aFullPNString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(0,aMessage,'Error!')


	def update( self ):
        
		aString = str( self.theFullPN()[ID] )
		aString += ':\n' + str( self.theFullPN()[PROPERTY] )        
		self.theIDEntry.set_text  ( aString )

		aValue = self.theSession.theSimulator.getEntityProperty( createFullPNString( self.theFullPN() ) )
		
		self.theActualValue = aValue
		self.theBarLength , self.theMultiplier , self.thePositiveFlag \
		                              = self.calculateBarLength( aValue )

		aIndicator = (aValue / (float)(10**(self.theMultiplier))) \
		                              * self.thePositiveFlag

		self['progressbar'].set_value(int(self.theBarLength))
		self['progressbar'].set_format_string(str(aValue))


		self.theMultiplier1Entry.set_text(str(int(self.theMultiplier-1)))
		self['multiplier_entry'].set_text(str(int(self.theMultiplier+2)))


	def updateByAuto( self, aValue ):

		self.theFixFlag = 0
		self.update()


	def updateByIncrease( self , obj ):

		self.theFixFlag = TRUE
		self['fix_checkbutton'].set_active( TRUE )
		aNumberString =  self['multiplier_entry'].get_text()

		try:
			aNumber = string.atof( aNumberString )
		except:
			anErrorMessage = "Numeric charactor must be inputted!"
			aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,'Error !')
			return None

		aNumber = aNumber + 1

		self.pull = aNumber

		self.update()


	def updateByDecrease( self,obj ):

		self.theFixFlag = TRUE
		self['fix_checkbutton'].set_active( TRUE )
		aNumberString =  self['multiplier_entry'].get_text()
		aNumber = string.atof( aNumberString )
		aNumber = aNumber - 1
		self.pull = aNumber
		self.update()


	def updateByTextentry(self, obj):

		aNumberString = obj.get_text()

		self['fix_checkbutton'].set_active( TRUE )

		try:
			aNumber = string.atof( aNumberString )
		except:
			anErrorMessage = "Numeric charactor must be inputted!"
			aWarningWindow = ConfirmWindow(OK_MODE,anErrorMessage,'Error !')
			return None

		self.theFixFlag = TRUE

		self.pull = aNumber
		self.update()


	def updateByFix(self, autobutton):
		self.theFixFlag = self['fix_checkbutton'].get_active()
		self.update()


	def calculateBarLength( self, value ):

		if value < 0 :
			value = - value
			aPositiveFlag = -1
		else :
			aPositiveFlag = 1

		#if self['fix_checkbutton'].get_active() :
		if self.theFixFlag == TRUE:

			aMultiplier = self.pull-2
		else :

			if value == 0 :
				aMultiplier = 2
			else :
				aMultiplier = (int)(Numeric.log10(value))
			self.pull = aMultiplier+2

		if value == 0:
			aBarLength = 0
		else :
			aBarLength = (Numeric.log10(value)+1-aMultiplier)*1000/3

		return  aBarLength, aMultiplier, aPositiveFlag
                

	def changeValue( self, value ):
		self.updateByAuto( value )

   
if __name__ == "__main__":

    class simulator:

        dic={('Variable','/CELL/CYTOPLASM','ATP','value') : (1950,),}
        
        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]
        
        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Variable','/CELL/CYTOPLASM','ATP','value')

    def mainQuit( obj, data ):
        gtk.mainquit()

    def mainLoop():
        # FIXME: should be a custom function
        gtk.mainloop()

    def main():
        aPluginManager = Plugin.PluginManager()
        aBargraphWindow = BargraphWindow( 'plugins', simulator(), [fpn,], aPluginManager )
        aBargraphWindow.addHandler( 'gtk_main_quit', mainQuit )
        
        mainLoop()


    # propertyValue1 = -750.0000

    main()
