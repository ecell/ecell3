AUTOMAKE_OPTIONS = foreign

SUBDIRS = glade doc plugins  # plugin's'!

EXTRA_DIST = \
preferences.ini\
AboutModelEditor.py\
AutoLayout.py\
AutosaveWindow.py\
Buffer.py\
BufferFactory.py\
ClassEditor.py\
ClassEditorWindow.py\
ClassList.py\
ClassPropertyList.py\
Clipboard.py\
Command.py\
CommandMultiplexer.py\
CommandQueue.py\
ComplexLine.py\
ComplexShape.py\
ConfirmWindow.py\
ConnectionObject.py\
ConnectionObjectEditorWindow.py\
Constants.py\
DMInfo.py\
EditorObject.py\
EntityCommand.py\
EntityEditor.py\
EntityList.py\
EntityListTab.py\
Error.py\
FullIDBrowserWindow.py\
GraphicalUtils.py\
Layout.py\
LayoutBuffer.py\
LayoutBufferFactory.py\
LayoutCommand.py\
LayoutEml.py\
LayoutManager.py\
LayoutXML.py\
LineDescriptor.py\
LinePropertyComponent.py\
LinePropertyEditor.py\
ListWindow.py\
MEMainWindow.py\
MEVariableReferenceEditor.py\
ModelEditor.py\
ModelStore.py\
MultiLineEditor.py\
NestedListEditor.py\
ObjectEditorWindow.py\
PackingStrategy.py\
PathwayCanvas.py\
PathwayEditor.py\
PopupMenu.py\
ProcessObject.py\
PropertyEditor.py\
PropertyList.py\
ResizeableText.py\
RunTester.py\
Runtime.py\
ShapeDescriptor.py\
ShapePluginManager.py\
ShapePropertyComponent.py\
StepperChooser.py\
StepperCommand.py\
StepperEditor.py\
StepperList.py\
StepperTab.py\
SystemObject.py\
SystemTree.py\
TextObject.py\
Utils.py\
VariableObject.py\
VariableReferenceEditorComponent.py\
ViewComponent.py\
Window.py\
config.py

EXTRA_DIST +=\
ecell3-model-editor

all-local:
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py config
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py build

mostlyclean-local:
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py clean

install-exec-local:
	DESTDIR=$(DESTDIR); \
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py install -f --prefix=$(PREFIX) $${DESTDIR:+--root }$${DESTDIR}


