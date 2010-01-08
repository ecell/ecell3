from ecell.FullID import *
from ecell.ecssupport import *
from ConfirmWindow import *

ConfirmWindow( "Loading model, creating loggers", 'Test' )

loadModel( 'Drosophila.eml' )

fpn_p0 = createFullPN( 'Variable:/CELL/CYTOPLASM:P0:Value' )
fpn_p1 = createFullPN( 'Variable:/CELL/CYTOPLASM:P1:Value' )



log_p0 = createLoggerStub('Variable:/CELL/CYTOPLASM:P0:Value' )
log_p0.setLoggerPolicy( [10,0,0,0] )
log_p0.create()

ConfirmWindow( "Logger for Variable P0 created", 'Test' )

run(10)

ConfirmWindow( "Simulation run for 10 secs", "Test" )



# save model
saveModel( 'Drosophila2.eml')

ConfirmWindow( "Model saved", 'Test' )



#create entitylistwindow
entitylist1 = createEntityListWindow( )

ConfirmWindow( "EntityListWindow tested", 'Test' )


#create and display StepperWindow
stepper1=openWindow('StepperWindow')

#here comes stepperwindow specific code

ConfirmWindow( "StepperWindow tested", 'Test' )


#create plugins on board
ConfirmWindow( "Creating boardwindow", 'Test' )

board=openWindow('BoardWindow')
board.setPackDirectionForward(True)
board.setTableSize(1)

ConfirmWindow( "Creating pluginwindows on board", 'Test' )

board_plugin1=createPluginOnBoard( 'BargraphWindow', [ fpn_p0 , fpn_p1 ] )
board_plugin2=createPluginOnBoard( 'TracerWindow', [ fpn_p0 , fpn_p1 ] )
board_plugin3=createPluginOnBoard( 'PropertyWindow', [ fpn_p1 ] )
board_plugin4=createPluginOnBoard( 'TracerWindow', [ fpn_p1 ] )

ConfirmWindow( "Changing pack direction", 'Test' )

board.setTableSize(2)
board.setPackDirectionForward(True)
board.deletePluginWindowByTitle(board_plugin1.getTitle())

#here comes boardwindow specific code

ConfirmWindow( "BoardWindow tested", 'Test' )

board.close()

ConfirmWindow( "BoardWindow closed", 'Test' )

ConfirmWindow( "Creating standalone pluginwindows", 'Test' )

#create standalone pluginwindows
tracer1=createPluginWindow( 'TracerWindow', [ fpn_p0, fpn_p1 ] )
variable1=createPluginWindow( 'VariableWindow', [ fpn_p0 ] )
property1=createPluginWindow( 'PropertyWindow', [ fpn_p1 ] )
bargraph1=createPluginWindow( 'BargraphWindow', [ fpn_p1 ] )


tracer1.present()

ConfirmWindow( "Change tracerwindow to strip mode", 'Test' )
tracer1.showStrip()
tracer1.setStripInterval( 500)
run( 2000 )

ConfirmWindow( "Hiding a trace", 'Test' )

tracer1.setTraceVisible('Variable:/CELL/CYTOPLASM:P0:Value', False)

ConfirmWindow( "Unhiding a trace", 'Test' )

tracer1.setTraceVisible('Variable:/CELL/CYTOPLASM:P0:Value', True)


ConfirmWindow( "Change to history mode", 'Test' )
tracer1.showHistory()

ConfirmWindow( "Zooming In", 'Test' )
tracer1.zoomIn( 150, 1500, 500000, 900000)
ConfirmWindow( "Zooming In again", 'Test' )
tracer1.zoomIn( 700, 750, 600000, 800000)

ConfirmWindow( "Change scale of tracerwindow", 'Test' )

tracer1.setScale( "Vertical", "Log10" ) #set scale to log10
tracer1.setScale( "Horizontal", "Log10" )

ConfirmWindow( "Zooming Out", 'Test' )
tracer1.zoomOut(1)

# change to plot
ConfirmWindow( "Show phase plot", 'Test' )
tracer1.setXAxis( fpn_p0 )

ConfirmWindow( "Resize tracer", 'Test' )

tracer1.resize(800,600)

ConfirmWindow( "Show time plot again", "Test" )
tracer1.setXAxis( "Time" )

ConfirmWindow( "TracerWindow tested", 'Test' )


# by interval
saveLoggerData( createFullPNString( fpn_p0 ), aSaveDirectory='.', anInterval=0.2, aStartTime=500, anEndTime=1000 )

# all data
saveLoggerData( createFullPNString( fpn_p1 ) , aSaveDirectory='.' )

ConfirmWindow( "Logger data files saved", 'Test' )



if ConfirmWindow( "Quit application?", 'Test' ):
	QuitGUI()
else:
    # show mainwindow
    # give control to user
    openWindow( 'MainWindow' )
    GUI_interact()
