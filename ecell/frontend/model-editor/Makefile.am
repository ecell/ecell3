SUBDIRS = glade plugins doc

ECELL_STAGING_HOME=$(abs_top_builddir)/tmp
PYECELL_STAGING_HOME=$(ECELL_STAGING_HOME)/pyecell

PYTHON_ = env PYTHONPATH=$(PYECELL_STAGING_HOME) sysconfdir=$(sysconfdir) datarootdir=$(datarootdir) libdir=$(libdir) $(PYTHON)

EXTRA_DIST = \
	ecell/ui/model_editor/__init__.py\
	ecell/ui/model_editor/AboutModelEditor.py\
	ecell/ui/model_editor/AutoLayout.py\
	ecell/ui/model_editor/AutosaveWindow.py\
	ecell/ui/model_editor/Buffer.py\
	ecell/ui/model_editor/BufferFactory.py\
	ecell/ui/model_editor/ClassEditor.py\
	ecell/ui/model_editor/ClassEditorWindow.py\
	ecell/ui/model_editor/ClassList.py\
	ecell/ui/model_editor/ClassPropertyList.py\
	ecell/ui/model_editor/Clipboard.py\
	ecell/ui/model_editor/Command.py\
	ecell/ui/model_editor/CommandMultiplexer.py\
	ecell/ui/model_editor/CommandQueue.py\
	ecell/ui/model_editor/ComplexLine.py\
	ecell/ui/model_editor/ComplexShape.py\
	ecell/ui/model_editor/Config.py\
	ecell/ui/model_editor/ConfirmWindow.py\
	ecell/ui/model_editor/ConnectionObject.py\
	ecell/ui/model_editor/ConnectionObjectEditorWindow.py\
	ecell/ui/model_editor/Constants.py\
	ecell/ui/model_editor/DMInfo.py\
	ecell/ui/model_editor/EditorObject.py\
	ecell/ui/model_editor/EntityCommand.py\
	ecell/ui/model_editor/EntityEditor.py\
	ecell/ui/model_editor/EntityList.py\
	ecell/ui/model_editor/EntityListTab.py\
	ecell/ui/model_editor/Error.py\
	ecell/ui/model_editor/FileSelection.py \
	ecell/ui/model_editor/FullIDBrowserWindow.py\
	ecell/ui/model_editor/GraphicalUtils.py\
	ecell/ui/model_editor/Layout.py\
	ecell/ui/model_editor/LayoutBuffer.py\
	ecell/ui/model_editor/LayoutBufferFactory.py\
	ecell/ui/model_editor/LayoutCommand.py\
	ecell/ui/model_editor/LayoutEml.py\
	ecell/ui/model_editor/LayoutManager.py\
	ecell/ui/model_editor/LayoutXML.py\
	ecell/ui/model_editor/LineDescriptor.py\
	ecell/ui/model_editor/LinePropertyComponent.py\
	ecell/ui/model_editor/LinePropertyEditor.py\
	ecell/ui/model_editor/ListWindow.py\
	ecell/ui/model_editor/MEMainWindow.py\
	ecell/ui/model_editor/MEVariableReferenceEditor.py\
	ecell/ui/model_editor/ModelEditor.py\
	ecell/ui/model_editor/ModelStore.py\
	ecell/ui/model_editor/MultiLineEditor.py\
	ecell/ui/model_editor/NestedListEditor.py\
	ecell/ui/model_editor/ObjectEditorWindow.py\
	ecell/ui/model_editor/PackingStrategy.py\
	ecell/ui/model_editor/PathwayCanvas.py\
	ecell/ui/model_editor/PathwayEditor.py\
	ecell/ui/model_editor/PopupMenu.py\
	ecell/ui/model_editor/ProcessObject.py\
	ecell/ui/model_editor/PropertyEditor.py\
	ecell/ui/model_editor/PropertyList.py\
	ecell/ui/model_editor/ResizeableText.py\
	ecell/ui/model_editor/RunTester.py\
	ecell/ui/model_editor/Runtime.py\
	ecell/ui/model_editor/ShapeDescriptor.py\
	ecell/ui/model_editor/ShapePluginManager.py\
	ecell/ui/model_editor/ShapePropertyComponent.py\
	ecell/ui/model_editor/StepperChooser.py\
	ecell/ui/model_editor/StepperCommand.py\
	ecell/ui/model_editor/StepperEditor.py\
	ecell/ui/model_editor/StepperList.py\
	ecell/ui/model_editor/StepperTab.py\
	ecell/ui/model_editor/SystemObject.py\
	ecell/ui/model_editor/SystemTree.py\
	ecell/ui/model_editor/TextObject.py\
	ecell/ui/model_editor/Utils.py\
	ecell/ui/model_editor/VariableObject.py\
	ecell/ui/model_editor/VariableReferenceEditorComponent.py\
	ecell/ui/model_editor/ViewComponent.py\
	ecell/ui/model_editor/Window.py\
	ecell/ui/model_editor/opengui.py

EXTRA_DIST +=\
	model-editor.ini\
	setup.py

EXTRA_DIST +=\
	ecell3-model-editor

all-local:
	$(PYTHON_) setup.py config
	$(PYTHON_) setup.py build

mostlyclean-local:
	rm -rf build

install-exec-local:
	$(PYTHON_) setup.py install -f --prefix=$(PREFIX) --install-scripts=$(bindir) $(if $(DESTDIR),--root $(DESTDIR), )
