#!/usr/bin/env python2

import string

import gtk
import gnome.ui
import GDK
import libglade
import Numeric


class Window:

    def __init__( self, gladefile=None, root=None ):
        self.widgets = libglade.GladeXML( filename=gladefile, root=root )

    def addHandlers( self, handlers ):
        self.widgets.signal_autoconnect( handlers )
        
    def addHandler( self, name, handler, *args ):
        self.widgets.signal_connect( name, handler, args )

    def getWidget( self, key ):
        return self.widgets.get_widget( key )

    def __getitem__( self, key ):
        return self.widgets.get_widget( key )


class MainWindow(Window):

    def __init__( self, gladefile ):

        self.thePositiveFlag = 1
        self.theAutoChangeFlag = 1

        self.theActualValue = 0
        self.theBarLength = 0
        self.theMultiplier = 0

        self.theHandlerMap = {
                              'level_spinbutton_activate': self.updateBySpinbutton,
                              'level_spinbutton_changed': self.updateBySpinbutton,
                              'auto_button_toggled': self.updateByAutoButton
                              }

        Window.__init__( self, gladefile )
        self.addHandlers( self.theHandlerMap )

    def updateByAuto( self, value ):
        self.theAutoChangeFlag = 1
        self.update( value )

    def updateBySpinbutton( self, spinbutton_obj ):
        if self.theAutoChangeFlag :
            pass
        else :
            self['auto_button'].set_active( 0 )
        self.update( self.theActualValue )

        aNumberString =  spinbutton_obj.get_text()
        aNumber = string.atof( aNumberString )
        self['level_spinbutton'].set_value(aNumber)
        # value = propertyValue1

        self.theAutoChangeFlag = 0

    def updateByAutoButton(self, autobutton):
        self.update( self.theActualValue)

    def calculateBarLength( self, value ):
        if value < 0 :
            value = - value
            aPositiveFlag = -1
        else :
            aPositiveFlag = 1

        if self['auto_button'].get_active() :
            aMultiplier = (int)(Numeric.log10(value))
        else :
            aMultiplier = self['level_spinbutton'].get_value()

#        aBarLength = (value / (float)(10**(aMultiplier)))
        aBarLength = (Numeric.log10(value)-aMultiplier+1)*10/3

        return  aBarLength, aMultiplier, aPositiveFlag
        
    def update( self, value ):
        self.theActualValue = value
        self.theBarLength , self.theMultiplier , self.thePositiveFlag \
                          = self.calculateBarLength( value )

#        aIndicator = self.theBarLength * self.thePositiveFlag
        aIndicator = (value / (float)(10**(self.theMultiplier))) \
                     * self.thePositiveFlag
        
        self['progressbar'].set_value(int(self.theBarLength))
        self['progressbar'].set_format_string(str(aIndicator))
        self['level_spinbutton'].set_value(self.theMultiplier)
        
    ### for test
    def changeValueFromEntryWindow( self, obj, a):
        
        aValueString = obj.get_text()
        aValue = string.atof( aValueString )
        print aValue
        self.changeValue( aValue )

    def changeValue( self, value ):
        self.updateByAuto( value )

def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()

def main():
    systemPath = '/CELL/CYTOPLASM'
    #    ID = 'ATPase: Activity'
    ID = 'ATPase'
    FQPI = systemPath + ':' + ID  
    
    aWindow = MainWindow( 'ProgressbarWindow.glade' )

    aWindow.addHandler( 'gtk_main_quit', mainQuit )
    aWindow.addHandler( 'on_entry1_activate', aWindow.changeValueFromEntryWindow)

    aWindow['property_id_label'].set_text(ID)

    aWindow.update(propertyValue1)

    # aWindow.setAuto("auto_button")
    #    aWindow.setLabel("frame1", propertyName)
    
    mainLoop()

if __name__ == "__main__":
    
    propertyValue1 = -750.0000
    main()


