from ecell.FullID import *
from ecell.ecssupport import *

loadModel( 'Drosophila.eml' )
if doesExist('MainWindow'):
	print 'Mainwindow exists'
else:
	print "MainWindow doesn't exist"



fpn_p0=createFullPN( 'Variable:/CELL/CYTOPLASM:P0:Value' )
fpn_p1=createFullPN( 'Variable:/CELL/CYTOPLASM:P1:Value' )
logger_min = getParameter('logger_min_interval')
log_p0 = createLoggerStub('Variable:/CELL/CYTOPLASM:P0:Value' )
log_p0.create()
logger_min = 1

setParameter('logger_min_interval',logger_min)

run()

#test mainwindow
#MainWindow has been automatically created previously if run() was called
if doesExist('MainWindow'):
	print 'Mainwindow exists'
	mw=getWindow('MainWindow')
else:
	print "MainWindow doesn't exist"
	mw=openWindow('MainWindow')

#here comes mainwindow specific code
mw.hideMessageWindow()
mw.setStepSize(0.1)
mw.setStepType(True)
mw.stepSimulation()
openConfirmWindow( "MessageWindow hidden, stepped", 'Test' )
mw.showMessageWindow()
mw.resize(400,500)
mw.move(200,200)
mw.startSimulation()
print mw.getStepType()
print mw.getStepSize()
openConfirmWindow( "MainWindow tested", 'Test' )
mw.close()
#create entitylistwindow
entitylist1 = createEntityListWindow( )
entitylist2 = createEntityListWindow( )
print "created 2 windows"

entitylist1.listEntity('Process' , ('System','/CELL','CYTOPLASM'))
print "entity listed"

#here comes entitywindow specific code

deleteEntityListWindow( entitylist2 )

openConfirmWindow( "EntityListWindow tested", 'Test' )
entitylist1.iconify()

#create and display LoggerWindow
displayWindow('LoggerWindow')
logger1=getWindow('LoggerWindow')

#here comes loggerwindow specific code

logger1.saveDataFile(fpn_p0, aDirectory='/home/gabor/data', anInterval=0.2, aStartTime=10, anEndTime=20, fileType='ecd')

print logger1.getDataFileType()

print logger1.getDirectory()

print logger1.getEndTime()

print logger1.getInterval()

print logger1.getStartTime()

print logger1.getUseDefaultInterval()

print logger1.getUseDefaultTime()

logger1.resetAllValues( obj=None)

logger1.setDataFileType( 'ecd' )

logger1.setDirectory( '/home/gabor/data2')

logger1.setEndTime( 400 )

logger1.setInterval(1)

logger1.setStartTime(100)

logger1.setUseDefaultInterval(True)

logger1.setUseDefaultTime(True)

logger1.saveDataFile(fpn_p1)

openConfirmWindow( "LoggerWindow tested", 'Test' )

#create and display StepperWindow
stepper1=openWindow('StepperWindow')

#here comes stepperwindow specific code

stepper1.selectStepperID('DE')

stepper1.selectProperty('MinStepInterval')

stepper1.updateProperty(1e-49)

openConfirmWindow( "StepperWindow tested", 'Test' )


#create plugins on board
board=openWindow('BoardWindow')
board.resize(500,500)
board.move(200,200)
board.setPackDirectionForward(True)
board.setTableSize(1)
board_plugin1=createPluginOnBoard( 'DigitalWindow', [ fpn_p0 , fpn_p1 ] )
board_plugin2=createPluginOnBoard( 'TracerWindow', [ fpn_p0 , fpn_p1 ] )
board_plugin3=createPluginOnBoard( 'PropertyWindow', [ fpn_p1 ] )
board_plugin4=createPluginOnBoard( 'TracerWindow', [ fpn_p1 ] )

print board_plugin4.isStandAlone()

openConfirmWindow( "changing pack direction", 'Test' )
board.setTableSize(2)
board.setPackDirectionForward(True)
board.deletePluginWindowByTitle(board_plugin1.getTitle())

#here comes boardwindow specific code

openConfirmWindow( "BoardWindow tested", 'Test' )
board.iconify()

#create standalone pluginwindows
tracer1=createPluginWindow( 'TracerWindow', [ fpn_p0, fpn_p1 ] )
variable1=createPluginWindow( 'VariableWindow', [ fpn_p0 ] )
property1=createPluginWindow( 'PropertyWindow', [ fpn_p1 ] )
bargraph1=createPluginWindow( 'BargraphWindow', [ fpn_p1 ] )

#here comes pluginwindow specific code

print bargraph1.isStandAlone()
tracer1.iconify()
variable1.resize(400,400)
property1.close()
bargraph1.move(100,100)
bargraph1.editTitle('Bargraph %s' %createFullPNString(bargraph1.theFullPN()))
tracer1.present()
tracer1.hideGUI()
tracer1.resize( 600,400)
tracer1.setScale( False ) #set scale to log10
tracer1.setStripInterval( 500)
print "tracer settracevisible"

tracer1.setTraceVisible('Variable:/CELL/CYTOPLASM:P0:Value', False)
openConfirmWindow( "TracerWindow tested", 'Test' )
tracer1.smallSize()
setUpdateInterval(20)
run( 1000 )

openConfirmWindow( "Pluginwindows tested", 'Test' )
tracer1.logAll()
tracer1.showHistory()
tracer1.setTraceVisible('Variable:/CELL/CYTOPLASM:P0:Value', True)

run(1000)
tracer1.largeSize()

# zoom
tracer1.zoomIn( 150, 200, 0.1, 2.5)

tracer1.zoomIn( 170, 175, 0.15, 2.15)

tracer1.zoomIn( 172, 173, 0.75, 1.15)

tracer1.zoomOut(1)

openConfirmWindow( "TracerWindow tested", 'Test' )
tracer1.showStrip()

#create and display InterfaceWindow
interface1=openWindow('InterfaceWindow')

#here comes interfacewindow specific code
interface1.selectPlugin(property1.getTitle())

openConfirmWindow( "InterfaceWindow tested", 'Test' )

#hand over control to MainWindow
openWindow('MainWindow')
#GUI_interact()

print "script has run"
board.close()
tracer1.close()
if openConfirmWindow( "Quit application?", 'Test' ):
	QuitGUI()

