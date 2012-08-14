import gtk
import os

__all__ = [
    'FileSelection',
    ]

class FileSelection(gtk.FileChooserDialog):
    actions = {
        'open': gtk.FILE_CHOOSER_ACTION_OPEN,
        'save': gtk.FILE_CHOOSER_ACTION_SAVE,
        'select_folder': gtk.FILE_CHOOSER_ACTION_SELECT_FOLDER,
        'create_folder': gtk.FILE_CHOOSER_ACTION_CREATE_FOLDER,
        }

    def __init__(self, *arg, **kwarg):
        gtk.FileChooserDialog.__init__(self, *arg, **kwarg)
        self.set_current_folder(os.getcwd())
        self.cancel_button = self.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
        self.ok_button = self.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)
        self.ok_button.grab_default()
        if hasattr(self, "set_alternative_button_order"):
            self.set_alternative_button_order([gtk.RESPONSE_OK, gtk.RESPONSE_CANCEL])

    def __set_action(self, action):
        gtk.FileChooserDialog.set_action(self, self.actions[action])

    def __get_action(self):
        a = gtk.FileChooserDialog.get_action(self)
        for k, v in actions.iteritems():
            if a == v:
                return k
        return None

    action = property(__get_action, __set_action)

    def show_fileop_buttons(self):
        pass

    def complete(self, pattern):
        filter = gtk.FileFilter()
        filter.add_pattern(pattern)
        self.set_filter(filter)
