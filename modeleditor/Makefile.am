AUTOMAKE_OPTIONS = foreign

SUBDIRS = glade doc


modeleditordir=${pkglibdir}
modeleditor_PYTHON=\
preferences.ini\
AboutModelEditor.py\
AutosaveWindow.py\
Buffer.py\
BufferFactory.py\
Command.py\
CommandQueue.py\
ComplexShape.py\
ComplexLine.py\
ConfirmWindow.py\
ConnectionObject.py\
ConnectionObjectEditorWindow.py\
Constants.py\
DMInfo.py\
EditorObject.py\
EntityCommand.py\
EntityEditor.py\
EntityList.py\
EntityListWindow.py\
Error.py\
FullIDBrowserWindow.py\
Layout.py\
LayoutBuffer.py\
LayoutBufferFactory.py\
LayoutCommand.py\
LayoutEml.py\
LayoutManager.py\
LayoutManagerWindow.py\
LayoutXML.py\
LinePropertyEditor.py\
LineDescriptor.py\
ListWindow.py\
MainWindow.py\
ModelEditor.py\
ModelStore.py\
MultiLineEditor.py\
NestedListEditor.py\
ObjectEditorWindow.py\
PathwayEditor.py\
PopupMenu.py\
ProcessObject.py\
PropertyEditor.py\
PropertyList.py\
ResizeableText.py\
ShapePropertyComponent.py\
StepperChooser.py\
StepperCommand.py\
StepperEditor.py\
StepperList.py\
StepperWindow.py\
SystemObject.py\
SystemTree.py\
TextObject.py\
Utils.py\
VariableObject.py\
VariableReferenceEditor.py\
ViewComponent.py\
Window.py\
PathwayCanvas.py\
Config.py\
CommandMultiplexer.py\
PackingStrategy.py\
ecell3-model-editor\
opengui.py\
VariableReferenceEditorComponent.py\
ShapePropertyComponent.py\
ShapeDescriptor.py\
ObjectEditorWindow.py\
LinePropertyComponent.py\
LayoutManagerWindow.py\
GraphicalUtils.py\
ConnectionObjectEditorWindow.py\
ComplexShape.py


EXTRA_DIST=${modeleditor_PYTHON} ${BINS}

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../ecell/pyecs/windows/py2exeSetup
WIN_ICON = glade/modeleditor.ico
PATH_VARIABLES = ../ecell/pyecs/windows/pathVariables

BINS =\
ecell3-model-editor

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --gui py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd

