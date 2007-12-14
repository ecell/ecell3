SUBDIRS = glade doc plugins

ECELL_LIB_DIR = $(libdir)/$(ECELL_DIRNAME)
modeleditordir = $(ECELL_LIB_DIR)/model-editor

modeleditor_PYTHON = \
	ShapePluginManager.py \
	AboutModelEditor.py \
	AutoLayout.py \
	AutosaveWindow.py \
	Buffer.py \
	BufferFactory.py \
	ClassEditor.py \
	ClassList.py \
	ClassEditorWindow.py \
	ClassPropertyList.py \
	Clipboard.py \
	Command.py \
	CommandQueue.py \
	ComplexShape.py \
	ComplexLine.py \
	ConfirmWindow.py \
	ConnectionObject.py \
	ConnectionObjectEditorWindow.py \
	Constants.py \
	DMInfo.py \
	EditorObject.py \
	EntityCommand.py \
	EntityEditor.py \
	EntityList.py \
	EntityListTab.py \
	Error.py \
	FullIDBrowserWindow.py \
	Layout.py \
	LayoutBuffer.py \
	LayoutBufferFactory.py \
	LayoutCommand.py \
	LayoutEml.py \
	LayoutManager.py \
	LayoutXML.py \
	LinePropertyEditor.py \
	LineDescriptor.py \
	ListWindow.py \
	MEMainWindow.py \
	ModelEditor.py \
	ModelStore.py \
	MultiLineEditor.py \
	NestedListEditor.py \
	ObjectEditorWindow.py \
	PathwayEditor.py \
	PopupMenu.py \
	ProcessObject.py \
	PropertyEditor.py \
	PropertyList.py \
	ResizeableText.py \
	Runtime.py \
	ShapePropertyComponent.py \
	StepperChooser.py \
	StepperCommand.py \
	StepperEditor.py \
	StepperList.py \
	StepperTab.py \
	SystemObject.py \
	SystemTree.py \
	TextObject.py \
	Utils.py \
	VariableObject.py \
	MEVariableReferenceEditor.py \
	ViewComponent.py \
	Window.py \
	PathwayCanvas.py \
	Config.py \
	CommandMultiplexer.py \
	PackingStrategy.py \
	opengui.py \
	VariableReferenceEditorComponent.py \
	ShapePropertyComponent.py \
	ShapeDescriptor.py \
	ObjectEditorWindow.py \
	LinePropertyComponent.py \
	GraphicalUtils.py \
	ConnectionObjectEditorWindow.py \
	ComplexShape.py

modeleditor_DATA = \
	preferences.ini

bin_SCRIPTS = ecell3-model-editor

EXTRA_DIST = $(modeleditor_PYTHON) $(modeleditor_DATA) $(bin_SCRIPTS)

clean-local:
	$(RM) -rf build *.exe *.zip *.dll *.pyd

