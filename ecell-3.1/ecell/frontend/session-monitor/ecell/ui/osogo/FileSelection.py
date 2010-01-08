import gtk

__all__ = [
    'FileSelection',
    ]

class FileSelection(gtk.FileChooserDialog):
    def __init__(self, *arg, **kwarg):
        gtk.FileChooserDialog.__init__(self, *arg, **kwarg)
        self.cancel_button = self.add_button(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL)
        self.ok_button = self.add_button(gtk.STOCK_OK, gtk.RESPONSE_OK)
        self.ok_button.grab_default()
        self.set_alternative_button_order([gtk.RESPONSE_OK, gtk.RESPONSE_CANCEL])

    def complete(self, pattern):
        filter = gtk.FileFilter()
        filter.add_pattern(pattern)
        self.set_filter(filter)
