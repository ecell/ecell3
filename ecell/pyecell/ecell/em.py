#!/usr/local/bin/python
#
# $Id$ $Date$

"""
A system for processing Python as markup embedded in text.
"""

__program__ = 'empy'
__version__ = '1.5'
__url__ = 'http://www.alcyone.com/pyos/empy/'
__author__ = 'Erik Max Francis <max@alcyone.com>'
__copyright__ = 'Copyright (C) 2002 Erik Max Francis'
__license__ = 'GPL'


import getopt
import string
import cStringIO
import re
import sys
import types

StringIO = cStringIO
del cStringIO


FAILURE_CODE = 1
DEFAULT_PREFIX = '@'
INTERNAL_MODULE_NAME = 'empy'
SIGNIFICATOR_RE_STRING = r"@%(\S)+\s*(.*)$"


class EmpyError(Exception):
    
    """The base class for all empy errors."""

    pass

class DiversionError(EmpyError):

    """An error related to diversions."""

    pass

class FilterError(EmpyError):

    """An error related to filters."""

    pass

class ParseError(EmpyError):

    """A parse error occurred."""

    pass

class TransientParseError(ParseError):

    """A parse error occurred which may be resolved by feeding more data.
    Such an error reaching the toplevel is an unexpected EOF error."""

    pass

class StackUnderflowError(EmpyError):

    """A stack underflow."""

    pass

class CommandLineError(EmpyError):

    """An error triggered by errors in the command line."""

    pass


class MetaError(Exception):

    """A wrapper around a real Python exception for including a copy of
    the context."""
    
    def __init__(self, contexts, exc):
        Exception.__init__(self, exc)
        self.contexts = contexts
        self.exc = exc

    def __str__(self):
        backtrace = map(lambda x: "%s:%d" % (x.name, x.line), self.contexts)
        return "%s: %s (%s)" % (self.exc.__class__, self.exc, \
                                (string.join(backtrace, ', ')))


class Stack:
    
    """A simple stack that behave as a sequence (with 0 being the top
    of the stack, not the bottom)."""

    def __init__(self, seq=None):
        if seq is None:
            seq = []
        self.data = seq

    def top(self):
        """Access the top element on the stack."""
        try:
            return self.data[-1]
        except IndexError:
            raise StackUnderflowError, "stack is empty for top"
        
    def pop(self):
        """Pop the top element off the stack and return it."""
        try:
            return self.data.pop()
        except IndexError:
            raise StackUnderflowError, "stack is empty for pop"
        
    def push(self, object):
        """Push an element onto the top of the stack."""
        self.data.append(object)

    def clone(self):
        """Create a duplicate of this stack."""
        return self.__class__(self.data[:])

    def __nonzero__(self): return len(self.data) != 0
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[-(index + 1)]


class AbstractFile:
    
    """An abstracted file that, when buffered, will totally buffer the
    file, including even the file open."""

    def __init__(self, filename, mode='w', bufferedOutput=0):
        self.filename = filename
        self.mode = mode
        self.bufferedOutput = bufferedOutput
        if bufferedOutput:
            self.bufferFile = StringIO.StringIO()
        else:
            self.bufferFile = open(filename, mode)
        self.done = 0

    def __del__(self):
        self.close()

    def write(self, data):
        self.bufferFile.write(data)

    def writelines(self, data):
        self.bufferFile.writelines(data)

    def flush(self):
        self.bufferFile.flush()

    def close(self):
        if not self.done:
            self.commit()
            self.done = 1

    def commit(self):
        if self.bufferedOutput:
            file = open(self.filename, self.mode)
            file.write(self.bufferFile.getvalue())
            file.close()
        else:
            self.bufferFile.close()

    def abort(self):
        if self.bufferedOutput:
            self.bufferFile = None
        else:
            self.bufferFile.close()
            self.bufferFile = None
        self.done = 1


class Stream:
    
    """A wrapper around an (output) file object which supports
    diversions and filtering."""
    
    def __init__(self, file):
        self.file = file
        self.currentDiversion = None
        self.diversions = {}
        self.filter = file
        self.done = 0

    def __del__(self):
        self.close()

    def write(self, data):
        if self.currentDiversion is None:
            self.filter.write(data)
        else:
            self.diversions[self.currentDiversion].write(data)
    
    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self.filter.flush()

    def close(self):
        if not self.done:
            self.undivertAll(1)
            self.filter.close()
            self.done = 1

    def install(self, filter=None):
        """Install a new filter; None means no filter.  Handle all the
        special shortcuts for filters here."""
        if filter is None or filter == [] or filter == ():
            # Shortcuts for "no filter."
            self.filter = self.file
        else:
            if type(filter) in (types.ListType, types.TupleType):
                filterShortcuts = list(filter)
            else:
                filterShortcuts = [filter]
            filters = []
            # Run through the shortcut filter names, replacing them with
            # full-fledged instances of Filter.
            for filter in filterShortcuts:
                if filter == 0:
                    filters.append(NullFilter())
                elif type(filter) is types.FunctionType or \
                     type(filter) is types.BuiltinFunctionType or \
                     type(filter) is types.BuiltinMethodType or \
                     type(filter) is types.LambdaType:
                    filters.append(FunctionFilter(filter))
                elif type(filter) is types.StringType:
                    filters.append(StringFilter(filter))
                elif type(filter) is types.DictType:
                    raise NotImplementedError, \
                          "mapping filters not yet supported"
                else:
                    filters.append(filter) # assume it's a Filter
            if len(filters) > 1:
                # If there's more than one filter provided, chain them
                # together.
                lastFilter = None
                for filter in filters:
                    if lastFilter is not None:
                        lastFilter.attach(filter)
                    lastFilter = filter
                lastFilter.attach(self.file)
                self.filter = filters[0]
            else:
                # If there's only one filter, assume that it's alone or
                # it's part of a chain that has already been manually chained;
                # just find the end.
                filter = filters[0]
                thisFilter, lastFilter = filter, filter
                while thisFilter is not None:
                    lastFilter = thisFilter
                    thisFilter = thisFilter.sink
                lastFilter.attach(self.file)
                self.filter = filter

    def revert(self):
        """Reset any current diversions."""
        self.currentDiversion = None

    def divert(self, diversion):
        """Start diverting."""
        if diversion is None:
            raise DiversionError, "diversion name must be non-None"
        self.currentDiversion = diversion
        if not self.diversions.has_key(diversion):
            self.diversions[diversion] = StringIO.StringIO()

    def undivert(self, diversion, purgeAfterwards=0):
        """Undivert a particular diversion."""
        if diversion is None:
            raise DiversionError, "diversion name must be non-None"
        if self.diversions.has_key(diversion):
            strFile = self.diversions[diversion]
            self.filter.write(strFile.getvalue())
            if purgeAfterwards:
                self.purge(diversion)
        else:
            raise DiversionError, "nonexistent diversion: %s" % diversion

    def purge(self, diversion):
        """Purge the specified diversion."""
        if diversion is None:
            raise DiversionError, "diversion name must be non-None"
        if self.diversions.has_key(diversion):
            del self.diversions[diversion]
            if self.currentDiversion == diversion:
                self.currentDiversion = None

    def undivertAll(self, purgeAfterwards=1):
        """Undivert all pending diversions."""
        if self.diversions:
            self.revert() # revert before undiverting!
            diversions = self.diversions.keys()
            diversions.sort()
            for diversion in diversions:
                self.undivert(diversion)
                if purgeAfterwards:
                    self.purge(diversion)
            
    def purgeAll(self):
        """Eliminate all existing diversions."""
        if self.diversions:
            self.diversions = {}
        self.currentDiversion = None


class ProxyFile:

    """The proxy file object that is intended to take the place of
    sys.stdout.  The proxy can manage a stack of file objects it is
    writing to, and an underlying raw file object."""

    def __init__(self, rawFile):
        self.stack = Stack()
        self.raw = rawFile
        self.current = rawFile

    def push(self, file):
        self.stack.push(file)
        self.current = file

    def pop(self):
        result = self.stack.pop()
        if self.stack:
            self.current = self.stack[-1]
        else:
            self.current = self.raw
        return result

    def write(self, data):
        self.current.write(data)

    def writelines(self, lines):
        self.current.writelines(lines)

    def flush(self):
        self.current.flush()

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


class Filter:

    """An abstract filter."""

    def __init__(self):
        if self.__class__ is Filter:
            raise NotImplementedError
        self.sink = None
        self.done = 0

    def write(self, data):
        raise NotImplementedError

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self.sink.flush()

    def close(self):
        self.flush()
        self.sink.close()

    def attach(self, filter):
        """Attach a filter to this one."""
        self.sink = filter

    def _last(self):
        """Find the last filter in this chain; this is the one that needs
        to be attached to the output stream.  This method is needed by
        PseudoModule.setFilter to hook things up properly."""
        this, last = self, self
        while this is not None:
            last = this
            this = this.sink
        return last

class NullFilter(Filter):

    """A filter that never sends any output to its sink."""

    def write(self, data): pass

class FunctionFilter(Filter):

    """A filter that works simply by pumping its input through a
    function which maps strings into strings."""
    
    def __init__(self, function):
        Filter.__init__(self)
        self.function = function

    def write(self, data):
        self.sink.write(self.function(data))

class StringFilter(Filter):

    """A filter that takes a translation string (256 characters) and
    filters any incoming data through it."""

    def __init__(self, table):
        if not (type(table) == types.StringType and len(table) == 256):
            raise FilterError, "table must be 256-character string"
        Filter.__init__(self)
        self.table = table

    def write(self, data):
        self.sink.write(string.translate(data, self.table))


class Scanner:

    """A scanner has the ability to look ahead through a string for
    special characters, respecting Python quotes."""
    
    def __init__(self, data=None):
        self.data = None
        self.set(data)

    def set(self, data):
        """Start the scanner digesting a new batch of data."""
        self.data = data

    def read(self, i, count=1):
        """Read count chars starting from i."""
        if len(self.data) < i + count:
            raise TransientParseError, "need more data to read"
        else:
            return self.data[i:i + count]

    def check(self, i, archetype=None):
        """Scan for the next single or triple quote, with the specified
        archetype (if the archetype is present, only trigger on those types
        of quotes.  Return the found quote or None."""
        quote = None
        if self.data[i] in '\'\"':
            quote = self.data[i]
            if len(self.data) - i < 3:
                for j in range(i, len(self.data)):
                    if self.data[i] == quote:
                        return quote
                else:
                    raise TransientParseError, "need to scan for rest of quote"
            if self.data[i + 1] == self.data[i + 2] == quote:
                quote = quote * 3
        if quote is not None:
            if archetype is None:
                return quote
            else:
                if archetype == quote:
                    return quote
                elif len(archetype) < len(quote) and archetype[0] == quote[0]:
                    return archetype
                else:
                    return None
        else:
            return None

    def next(self, target, i=0, j=None, mandatory=0):
        """Scan from i to j for the next occurrence of one of the characters
        in the target string; optionally, make the scan mandatory."""
        if mandatory:
            assert j is not None
        quote = None
        if j is None:
            j = len(self.data)
        while i < j:
            newQuote = self.check(i, quote)
            if newQuote:
                if newQuote == quote:
                    quote = None
                else:
                    quote = newQuote
                i = i + len(newQuote)
            else:
                c = self.data[i]
                if quote:
                    if c == '\\':
                        i = i + 1
                else:
                    if c in target:
                        return i
                i = i + 1
        else:
            if mandatory:
                raise ParseError, "expecting %s, not found" % target
            else:
                raise TransientParseError, "expecting ending character"

    def complex(self, enter, exit, i=0, j=None):
        """Scan from i for an ending sequence, respecting entries and
        exits."""
        quote = None
        depth = 0
        if j is None:
            j = len(self.data)
        while i < j:
            newQuote = self.check(i, quote)
            if newQuote:
                if newQuote == quote:
                    quote = None
                else:
                    quote = newQuote
                i = i + len(newQuote)
            else:
                c = self.data[i]
                if quote:
                    if c == '\\':
                        i = i + 1
                else:
                    if c == enter:
                        depth = depth + 1
                    elif c == exit:
                        depth = depth - 1
                        if depth < 0:
                            return i
                i = i + 1
        else:
            raise TransientParseError, "expecting end of complex expression"

    def word(self, i=0):
        """Scan from i for a simple word."""
        self.dataLen = len(self.data)
        while i < self.dataLen:
            if not self.data[i] in Parser.IDENTIFIER_REST:
                return i
            i = i + 1
        else:
            raise TransientParseError, "expecting end of word"

    def phrase(self, i=0):
        """Scan from i for a phrase (e.g., 'word', 'f(a, b, c)', 'a[i]', or
        combinations like 'x[i](a)'."""
        # Find the word.
        i = self.word(i)
        while i < len(self.data) and self.data[i] in '([':
            enter = self.data[i]
            exit = Parser.ENDING_CHARS[enter]
            i = self.complex(enter, exit, i + 1) + 1
        return i
    
    def simple(self, i=0):
        """Scan from i for a simple expression, which consists of one 
        more phrases separated by dots."""
        i = self.phrase(i)
        self.dataLen = len(self.data)
        while i < self.dataLen and self.data[i] == '.':
            i = self.phrase(i)
        # Make sure we don't end with a trailing dot.
        while i > 0 and self.data[i - 1] == '.':
            i = i - 1
        return i

    def keyword(self, word, i=0, j=None, mandatory=0):
        prefix = word[0] # keywords start with a prefix
        if mandatory:
            assert j is not None
        k = -1
        while 1:
            k = string.find(self.data, word, k + 1, j)
            if k < 0:
                if mandatory:
                    raise ParseError, "missing keyword %s" % word
                else:
                    raise TransientParseError, "missing keyword %s" % word
            if k == 0 or (k > 0 and self.data[k - 1] != prefix):
                return k


class Parser:
    
    """The core parser which also manages the output file."""
    
    SECONDARY_CHARS = "#%([{)]}`: \t\v\n\\"
    IDENTIFIER_FIRST = '_abcdefghijklmnopqrstuvwxyz' + \
                       'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    IDENTIFIER_REST = '0123456789' + IDENTIFIER_FIRST + '.'
    ENDING_CHARS = {'(': ')', '[': ']', '{': '}'}

    IF_RE = re.compile(r"^if\s+(.*)$")
    WHILE_RE = re.compile(r"^while\s+(.*)$")
    FOR_RE = re.compile(r"^for\s+(\S+)\s+in\s+(.*)$")
    ELSE_SEPARATOR = '@else:'

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.buffer = ''
        self.first = 1 # have we not yet started?
        self.scanner = Scanner()

    def process(self, data):
        """Process data in the form of a 3-triple:  the data preceding the
        token, the token, and the data following the token (which may
        include more tokens).  If no token is found, then it will return
        (data, None, '')."""
        loc = string.find(data, self.interpreter.prefix)
        if loc < 0:
            # No token, end of data.
            return data, None, ''
        else:
            start, end = string.split(data, self.interpreter.prefix, 1)
            if not end:
                raise ParseError, "illegal trailing token"
            elif end[0] in self.interpreter.prefix + Parser.SECONDARY_CHARS:
                code, end = end[0], end[1:]
                return start, self.interpreter.prefix + code, end
            elif end[0] in Parser.IDENTIFIER_FIRST:
                return start, self.interpreter.prefix, end
            else:
                raise ParseError, "unrecognized token sequence"

    def feed(self, data):
        """Feed the parser more data."""
        self.buffer = self.buffer + data

    def once(self):
        """Feed the parser more data (if that argument is presented)
        and have it attempt to parse it by only performing _one_
        expansion.  If a parse error occurs which may be eliminated by
        parsing more data, a TransientParseError is raised; catch and
        dispose of this error if you intend to send more data.  Return
        true if an expansion was performed, otherwise false."""
        # UNIX support for scriptability:  If #! is the first thing we see
        # then replace it with the comment sequence.
        prefix = self.interpreter.prefix
        if self.first and self.buffer[:2] == '#!':
            self.buffer = prefix + '#' + self.buffer[2:]
        self.first = 0
        if self.buffer:
            start, token, self.buffer = self.process(self.buffer)
            self.interpreter.stream.write(start)
            if token is None:
                # No remaining tokens, just pass through.
                self.interpreter.stream.write(self.buffer)
                self.buffer = ''
                return 0
            elif token in (prefix + ' ', prefix + '\t', \
                           prefix + '\v', prefix + '\n'):
                # Whitespace/line continuation.
                pass
            elif token in (prefix + ')', prefix + ']', prefix + '}'):
                self.interpreter.stream.write(token[-1])
            elif token == prefix + '\\':
                try:
                    self.scanner.set(self.buffer)
                    code = self.scanner.read(0)
                    i = 1
                    result = None
                    if code in '()[]{}\'\"\\': # literals
                        result = code
                    elif code == '0': # NUL
                        result = '\x00'
                    elif code == 'a': # BEL
                        result = '\x07'
                    elif code == 'b': # BS
                        result = '\x08'
                    elif code == 'd': # decimal code
                        decimalCode = self.scanner.read(i, 3)
                        i = i + 3
                        result = chr(string.atoi(decimalCode, 10))
                    elif code == 'e': # ESC
                        result = '\x1b'
                    elif code == 'f': # FF
                        result = '\x0c'
                    elif code == 'h': # DEL
                        result = '\x7f'
                    elif code == 'n': # LF (newline)
                        result = '\x0a'
                    elif code == 'o': # octal code
                        octalCode = self.scanner.read(i, 3)
                        i = i + 3
                        result = chr(string.atoi(octalCode, 8))
                    elif code == 'r': # CR
                        result = '\x0d'
                    elif code == 's': # SP
                        result = ' '
                    elif code == 't': # HT
                        result = '\x09'
                    elif code == 'u': # unicode
                        raise NotImplementedError, "no Unicode support"
                    elif code == 'v': # VT
                        result = '\x0b'
                    elif code == 'x': # hexadecimal code
                        hexCode = self.scanner.read(i, 2)
                        i = i + 2
                        result = chr(string.atoi(hexCode, 16))
                    elif code == 'z': # EOT
                        result = '\x04'
                    elif code == '^': # control character
                        controlCode = string.upper(self.scanner.read(i))
                        i = i + 1
                        if controlCode >= '@' and controlCode <= '`':
                            result = chr(ord(controlCode) - 0x40)
                        else:
                            raise ParseError, "invalid escape control code"
                    else:
                        raise ParseError, "unrecognized escape code"
                    assert result is not None
                    self.interpreter.stream.write(result)
                    self.buffer = self.buffer[i:]
                except ValueError:
                    raise ParseError, "invalid numeric escape code"
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix:
                # "Simple expression" expansion.
                try:
                    self.scanner.set(self.buffer)
                    i = self.scanner.simple()
                    code, self.buffer = self.buffer[:i], self.buffer[i:]
                    result = self.evaluate(code)
                    if result is not None:
                        self.interpreter.stream.write(str(result))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '#':
                # Comment.
                if string.find(self.buffer, '\n') >= 0:
                    self.buffer = string.split(self.buffer, '\n', 1)[1]
                else:
                    self.buffer = token + self.buffer
                    raise TransientParseError, "comment expecting newline"
            elif token == prefix + '%':
                # Significator.
                if string.find(self.buffer, '\n') >= 0:
                    line, self.buffer = string.split(self.buffer, '\n', 1)
                    if not line:
                        raise ParseError, "significator must have nonblank key"
                    if line[0] in ' \t\v\n':
                        raise ParseError, "no whitespace between % and key"
                    result = string.split(line, None, 1)
                    while len(result) < 2:
                        result.append('')
                    key, value = result
                    key = string.strip(key)
                    value = string.strip(value)
                    self.significate(key, value)
                else:
                    self.buffer = token + self.buffer
                    raise TransientParseError, "significator expecting newline"
            elif token == prefix + '(':
                # Uber-expression evaluation.
                self.scanner.set(self.buffer)
                try:
                    z = self.scanner.complex('(', ')', 0)
                    try:
                        q = self.scanner.next('$', 0, z, 1)
                    except ParseError:
                        q = z
                    try:
                        i = self.scanner.next('?', 0, q, 1)
                        try:
                            j = self.scanner.next(':', i, q, 1)
                        except ParseError:
                            j = q
                    except ParseError:
                        i = j = q
                    testCode = self.buffer[:i]
                    thenCode = self.buffer[i + 1:j]
                    elseCode = self.buffer[j + 1:q]
                    catchCode = self.buffer[q + 1:z]
                    self.buffer = self.buffer[z + 1:]
                    try:
                        result = self.evaluate(testCode)
                        if thenCode:
                            if result:
                                result = self.evaluate(thenCode)
                            else:
                                if elseCode:
                                    result = self.evaluate(elseCode)
                                else:
                                    result = None
                    except SyntaxError:
                        # Don't catch syntax errors; let them through.
                        raise
                    except:
                        if catchCode:
                            result = self.evaluate(catchCode)
                        else:
                            raise
                    if result is not None:
                        self.interpreter.stream.write(str(result))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '`':
                # Repr evalulation.
                self.scanner.set(self.buffer)
                try:
                    i = self.scanner.next('`', 0)
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    self.interpreter.stream.write(repr(self.evaluate(code)))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + ':':
                # In-place expression substitution.
                self.scanner.set(self.buffer)
                try:
                    i = self.scanner.next(':', 0)
                    j = self.scanner.next(':', i + 1)
                    code = self.buffer[:i]
                    self.buffer = self.buffer[j + 1:]
                    self.interpreter.stream.write("@:%s:" % code)
                    try:
                        result = self.evaluate(code)
                        if result is not None:
                            self.interpreter.stream.write(str(result))
                    finally:
                        self.interpreter.stream.write(":")
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '[':
                # Conditional and repeated substitution.
                self.scanner.set(self.buffer)
                try:
                    i = self.scanner.complex('[', ']', 0)
                    expansion, self.buffer = \
                        self.buffer[:i], self.buffer[i + 1:]
                    self.substitute(expansion)
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '{':
                # Statement evaluation.
                self.scanner.set(self.buffer)
                try:
                    i = self.scanner.complex('{', '}', 0)
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    self.execute(code)
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix * 2:
                # A simple character doubling.  As a precaution, check this
                # last in case the prefix has been changed to one of the
                # secondary characters.  This will prevent literal prefixes
                # from appearing in markup, but at least everything else
                # will function as expected!
                self.interpreter.stream.write(prefix)
            else:
                raise ParseError, "unknown token: %s" % token
            return 1
        else:
            return 0

    def parse(self):
        """Empty out the buffer."""
        while self.buffer:
            self.once()

    def done(self):
        """Declare that this parsing session is over and that any unparsed
        data should be considered an error."""
        if self.buffer:
            try:
                self.parse()
            except TransientParseError:
                if self.buffer and self.buffer[-1] != '\n':
                    # The data is not well-formed, so add a @ and a newline,
                    # ensuring that any transient parsing problem will be
                    # resolved but without affecting parsing.
                    self.buffer = self.buffer + '@\n'
                # A TransientParseError thrown here is a real parse error.
                self.parse()
                self.buffer = ''

    def substitute(self, substitution):
        """Do a command substitution."""
        self.scanner.set(substitution)
        z = len(substitution)
        i = self.scanner.next(':', 0, z, 1)
        command = string.strip(substitution[:i])
        rest = substitution[i + 1:]
        match = Parser.IF_RE.search(command)
        if match is not None:
            # if P : ... [@else: ...]
            testCode, = match.groups()
            try:
                self.scanner.set(rest)
                k = self.scanner.keyword(Parser.ELSE_SEPARATOR,
                                         0, len(rest), 1)
                thenExpansion = rest[:k]
                elseExpansion = rest[k + len(Parser.ELSE_SEPARATOR):]
            except ParseError:
                thenExpansion = rest
                elseExpansion = ''
            if self.evaluate(testCode):
                result = self.interpreter.expand(thenExpansion)
                self.interpreter.stream.write(str(result))
            else:
                if elseExpansion:
                    result = self.interpreter.expand(elseExpansion)
                    self.interpreter.stream.write(str(result))
            return
        match = Parser.WHILE_RE.search(command)
        if match is not None:
            # while P : ...
            testCode, = match.groups()
            expansion = rest
            while 1:
                if not self.evaluate(testCode):
                    break
                result = self.interpreter.expand(expansion)
                self.interpreter.stream.write(str(result))
            return
        match = Parser.FOR_RE.search(command)
        if match is not None:
            # for X in S : ...
            iterator, sequenceCode = match.groups()
            sequence = self.evaluate(sequenceCode)
            expansion = rest
            for element in sequence:
                self.interpreter.globals[iterator] = element
                result = self.interpreter.expand(expansion)
                self.interpreter.stream.write(str(result))
            return
        # If we get here, we didn't find anything
        raise ParseError, "unknown substitution type"

    def evaluate(self, expression, locals=None):
        """Evaluate an expression."""
        if locals is None:
            context = self.interpreter.contexts.top()
            locals = context.locals
        sys.stdout.push(self.interpreter.stream)
        try:
            if locals is not None:
                return eval(expression, self.interpreter.globals, locals)
            else:
                return eval(expression, self.interpreter.globals)
        finally:
            sys.stdout.pop()

    def execute(self, statements, locals=None):
        """Execute a statement."""
        if locals is None:
            context = self.interpreter.contexts.top()
            locals = context.locals
        sys.stdout.push(self.interpreter.stream)
        try:
            if string.find(statements, '\n') < 0:
                statements = string.strip(statements)
            if locals is not None:
                exec statements in self.interpreter.globals, locals
            else:
                exec statements in self.interpreter.globals
        finally:
            sys.stdout.pop()

    def significate(self, key, value):
        """Declare a significator."""
        name = '__%s__' % key
        self.interpreter.globals[name] = value

    def __nonzero__(self): return len(self.buffer) != 0


class Context:
    
    """An interpreter context, which encapsulates a name, an input
    file object, and a parser object."""

    def __init__(self, name, input, parser, locals=None, line=0):
        self.name = name
        self.input = input
        self.parser = parser
        self.locals = locals
        self.line = line
        self.exhausted = 1


class PseudoModule:
    
    """A pseudomodule for the builtin empy routines."""

    __name__ = INTERNAL_MODULE_NAME

    # Constants.

    VERSION = __version__
    SIGNIFICATOR_RE_STRING = SIGNIFICATOR_RE_STRING

    # Types.
    
    Filter = Filter
    NullFilter = NullFilter
    FunctionFilter = FunctionFilter
    StringFilter = StringFilter

    def __init__(self, interpreter):
        self.interpreter = interpreter

    # Identification.

    def identify(self):
        """Identify the topmost context with a 2-tuple of the name and
        line number."""
        context = self.interpreter.contexts.top()
        return context.name, context.line

    def setName(self, name):
        """Set the name of the topmost context."""
        context = self.interpreter.contexts.top()
        context.name = name
        
    def setLine(self, line):
        """Set the name of the topmost context."""
        context = self.interpreter.contexts.top()
        context.line = line

    # Source manipulation.

    def include(self, fileOrFilename, locals=None):
        """Take another filename or file object and process it."""
        if type(fileOrFilename) is types.StringType:
            # Either it's a string representing a filename ...
            filename = fileOrFilename
            name = filename
            file = open(filename, 'r')
        else:
            # ... or a file object.
            file = fileOrFilename
            name = "<%s>" % str(file.__class__)
        self.interpreter.new(name, file, locals)

    def expand(self, data, locals=None):
        """Do an explicit expansion pass on a string."""
        return self.interpreter.expand(data, locals)

    def quote(self, data):
        """Quote the given string so that if it were expanded it would
        evaluate to the original."""
        scanner = Scanner(data)
        result = []
        i = 0
        try:
            j = scanner.next('@', i)
            result.append(data[i:j])
            result.append('@@')
            i = j + 1
        except TransientParseError:
            pass
        result.append(data[i:])
        return string.join(result, '')

    # Pseudomodule manipulation.

    def flatten(self, keys=None):
        """Flatten the contents of the pseudo-module into the globals
        namespace."""
        if keys is None:
            keys = self.__dict__.keys() + self.__class__.__dict__.keys()
        dict = {}
        for key in keys:
            # The pseudomodule is really a class instance, so we need to
            # fumble use getattr instead of simply fumbling through the
            # instance's __dict__.
            dict[key] = getattr(self, key)
        # Stomp everything into the globals namespace.
        self.interpreter.globals.update(dict)

    # Prefix.

    def getPrefix(self):
        """Get the current prefix."""
        return self.interpreter.prefix

    def setPrefix(self, prefix):
        """Set the prefix."""
        self.interpreter.prefix = prefix

    # Diversions.

    def stopDiverting(self):
        """Stop any diverting."""
        self.interpreter.stream.revert()

    def startDiversion(self, diversion):
        """Start diverting to the given diversion name."""
        self.interpreter.stream.divert(diversion)

    def playDiversion(self, diversion):
        """Play the given diversion and then purge it."""
        self.interpreter.stream.undivert(diversion, 1)

    def replayDiversion(self, diversion):
        """Replay the diversion without purging it."""
        self.interpreter.stream.undivert(diversion, 0)

    def purgeDiversion(self, diversion):
        """Eliminate the given diversion."""
        self.interpreter.stream.purge(diversion)

    def playAllDiversions(self):
        """Play all existing diversions and then purge them."""
        self.interpreter.stream.undivertAll(1)

    def replayAllDiversions(self):
        """Replay all existing diversions without purging them."""
        self.interpreter.stream.undivertAll(0)

    def purgeAllDiversions(self):
        """Purge all existing diversions."""
        self.interpreter.stream.purgeAll()

    def getCurrentDiversion(self):
        """Get the name of the current diversion."""
        return self.interpreter.stream.currentDiversion

    def getAllDiversions(self):
        """Get the names of all existing diversions."""
        diversions = self.interpreter.stream.diversions.keys()
        diversions.sort()
        return diversions
    
    # Filter.

    def resetFilter(self):
        """Reset the filter so that it does no filtering."""
        self.interpreter.stream.install(None)

    def nullFilter(self):
        """Install a filter that will consume all text."""
        self.interpreter.stream.install(0)

    def getFilter(self):
        """Get the current filter."""
        filter = self.interpreter.stream.filter
        if filter is self.interpreter.stream.file:
            return None
        else:
            return filter

    def setFilter(self, filter):
        """Set the filter."""
        self.interpreter.stream.install(filter)


class Interpreter:
    
    """Higher-level manipulation of a stack of parsers, with file/line
    context."""

    def __init__(self, output=None, globals=None, \
                 prefix=DEFAULT_PREFIX, flatten=0):
        # Set up the stream first, since the pseudomodule needs it.
        if output is None:
            output = sys.__stdout__
        self.output = output
        self.stream = Stream(output)
        # Set up the globals.
        if globals is None:
            pseudo = PseudoModule(self)
            globals = {pseudo.__name__: pseudo}
        self.globals = globals
        self.prefix = prefix
        self.contexts = Stack()
        if flatten:
            pseudo.flatten()
        # Finally, install a proxy stdout if one hasn't been already.
        self.installProxy()

    def __del__(self):
        self.finish()

    def installProxy(self):
        """Install a proxy if necessary."""
        if sys.stdout is sys.__stdout__:
            sys.stdout = ProxyFile(sys.__stdout__)

    def new(self, name, input, locals=None):
        """Create a new interpreter context (a new file object to process)."""
        parser = Parser(self)
        context = Context(name, input, parser, locals)
        self.contexts.push(context)

    def expand(self, data, locals=None):
        """Do an explicit expansion pack in a subordinate interpreter."""
        input = StringIO.StringIO(data)
        output = StringIO.StringIO()
        subordinate = Interpreter(output, self.globals, self.prefix)
        subordinate.new('<expand>', input, locals)
        subordinate.go()
        subordinate.stream.flush()
        expansion = output.getvalue()
        subordinate.finish()
        return expansion

    def step(self):
        """Read one line of the current input interactively (attempt
        to parse it); if no input remains, finalize the context and
        pop it.  Return true if code was successfully parsed."""
        context = self.contexts.top()
        if context.exhausted:
            # If the current context is exhausted, get then next line.
            line = context.input.readline()
            context.line = context.line + 1
            if not line:
                context.parser.done()
                self.contexts.pop()
                return 0
            context.parser.feed(line)
            context.exhausted = 0
        try:
            # If the context changes after an expansion, abort this step.
            while context.parser and context is self.contexts.top():
                context.parser.once()
            if not context.parser:
                context.exhausted = 1
        except TransientParseError:
            context.exhausted = 1
        return 1

    def go(self):
        """Read the whole file."""
        while self.contexts:
            self.step()

    def wrap(self, bufferedOutput=0, rawErrors=0, exitOnError=1):
        """Wrap around go, handling errors appropriately."""
        done = 0
        while not done:
            try:
                self.go()
                done = 1
            except KeyboardInterrupt, e:
                # Keyboard interrupts should ignore the exitOnError setting.
                if bufferedOutput:
                    self.output.abort()
                meta = self.meta(e)
                self.handle(meta)
                if rawErrors:
                    raise
                sys.exit(FAILURE_CODE)
            except Exception, e:
                # A standard exception (other than a keyboard interrupt).
                if bufferedOutput:
                    self.output.abort()
                meta = self.meta(e)
                self.handle(meta)
                if exitOnError:
                    if rawErrors:
                        raise
                    sys.exit(FAILURE_CODE)
            except:
                # If we get here, then it must be a string exception, 
                # so get the error info from the sys module.
                e = sys.exc_type
                if bufferedOutput:
                    self.output.abort()
                meta = self.meta(e)
                self.handle(meta)
                if exitOnError:
                    if rawErrors:
                        raise
                    sys.exit(FAILURE_CODE)

    def finish(self):
        """Declare this interpreting session over; close the stream file
        object."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def meta(self, exc=None):
        """Construct a MetaError for the interpreter's current state."""
        return MetaError(self.contexts.clone(), exc)

    def handle(self, meta):
        """Handle a MetaError."""
        first = 1
        for context in meta.contexts:
            if first:
                if meta.exc is not None:
                    desc = "error: %s: %s" % (meta.exc.__class__, meta.exc)
                else:
                    desc = "error"
            else:
                desc = "from this context"
            first = 0
            sys.stderr.write('%s:%s: %s\n' % \
                             (context.name, context.line, desc))


def usage():
    sys.stderr.write("""\
Usage: %s [options] <filenames, or '-' for stdin>...

Valid options:
  -v --version                 Print version and exit
  -h --help                    Print usage and exit
  -k --suppress-errors         Print errors when they occur but do not exit
  -p --prefix=<char>           Change prefix to something other than @
  -f --flatten                 Flatten the members of pseudmodule to start
  -r --raw --raw-errors        Show raw Python errors
  -o --output=<filename>       Specify file for output as write
  -a --append=<filename>       Specify file for output as append
  -P --preprocess=<filename>   Interpret empy file before main processing
  -I --import=<modules>        Import Python modules before processing
  -E --execute=<statement>     Execute Python statement before processing
  -F --execute-file=<filename> Execute Python file before processing
  -B --buffered-output         Fully buffer output (even open) with -o or -a

The following expansions are supported (where @ is the prefix):
  @# ... NL                    Comment; remove everything up to newline
  @ WHITESPACE                 Remove following whitespace; line continuation
  @\\ ESCAPE_CODE               A C-style escape sequence
  @@                           Literal @; @ is escaped (duplicated prefix)
  @( EXPRESSION )              Evaluate expression and substitute with str
  @( TEST ? THEN )             If test is true, evaluate then
  @( TEST ? THEN : ELSE )      If test is true, evaluate then, otherwise else
  @( TRY $ CATCH )             Expand try expression, or catch if it raises
  @ SIMPLE_EXPRESSION          Evaluate simple expression and substitute;
                               e.g., @x, @x.y, @f(a, b), @l[i], etc.
  @` EXPRESSION `              Evaluate expression and substitute with repr
  @: EXPRESSION : DUMMY :      Evaluates to @:...:expansion:
  @[ if E : CODE ]             Expand then code if expression is true
  @[ if E : CODE @else: CODE ] Expand then code if expression is true, or else
  @[ while E : CODE ]          Repeatedly expand code while expression is true
  @[ for X in S : CODE ]       Expand code for each element in sequence
  @{ STATEMENTS }              Statements are executed for side effects
  @%% KEY WHITESPACE VALUE NL   Significator form of __KEY__ = VALUE

Valid escape sequences are:
  @\\0                          NUL, null
  @\\a                          BEL, bell
  @\\b                          BS, backspace
  @\\dDDD                       three-digital decimal code DDD
  @\\e                          ESC, escape
  @\\f                          FF, form feed
  @\\h                          DEL, delete
  @\\n                          LF, linefeed, newline
  @\\oOOO                       three-digit octal code OOO
  @\\r                          CR, carriage return
  @\\s                          SP, space
  @\\t                          HT, horizontal tab
  @\\v                          VT, vertical tab
  @\\xHH                        two-digit hexadecimal code HH
  @\\z                          EOT, end of transmission
  @\\^X                         control character ^X

The %s pseudomodule contains the following attributes:
  VERSION                      String representing empy version
  SIGNIFICATOR_RE_STRING       Regular expression string matching significators
  interpreter                  Currently-executing interpreter instance
  identify()                   Identify top context as name, line
  setName(name)                Set the name of the current context
  setLine(line)                Set the line number of the current context
  Filter                       The base class for custom filters
  include(file, [locals])      Include filename or file-like object
  expand(string, [locals])     Explicitly expand string
  quote(string)                Quote prefixes in provided string
  flatten()                    Flatten module contents into globals namespace
  getPrefix()                  Get current prefix
  setPrefix(char)              Set new prefix
  stopDiverting()              Stop diverting; data now sent directly to output
  startDiversion(name)         Start diverting to given diversion
  playDiversion(name)          Recall diversion and then eliminate it
  replayDiversion(name)        Recall diversion but retain it
  purgeDiversion(name)         Erase diversion
  playAllDiversions()          Stop diverting and play all diversions in order
  replayAllDiversions()        Stop diverting and replay all diversions
  purgeAllDiversions()         Stop diverting and purge all diversions
  getFilter()                  Get current filter
  resetFilter()                Reset filter; no filtering
  nullFilter()                 Install null filter
  setFilter(filter)            Install new filter or filter chain

Notes: Whitespace immediately inside parentheses of @(...) are
ignored.  Whitespace immediately inside braces of @{...} are ignored,
unless ...  spans multiple lines.  Use @{ ... }@ to suppress newline
following expansion.  Simple expressions ignore trailing dots; `@x.'
means `@(x).'.  A #! at the start of a file is treated as a @#
comment.
""" % (sys.argv[0], INTERNAL_MODULE_NAME))

def invoke(args):
    """Run a standalone instance of an empy interpeter."""
    # Initialize the options.
    Output = None
    Globals = None
    RawErrors = 0
    ExitOnError = 1
    BufferedOutput = 0
    Preprocessing = []
    Prefix = '@'
    Flatten = 0
    # Parse the arguments.
    pairs, remainder = getopt.getopt(args, 'vhkp:fro:a:P:I:E:F:B', ['version', 'help', 'suppress-errors', 'prefix=', 'flatten', 'raw', 'raw-errors', 'output=' 'append=', 'preprocess=', 'import=', 'execute=', 'execute-file=', 'buffered-output'])
    for option, argument in pairs:
        if option in ('-v',  '--version'):
            sys.stderr.write("%s version %s\n" % (__program__, __version__))
            return
        elif option in ('-h', '--help'):
            usage()
            return
        elif option in ('-k', '--suppress-errors'):
            ExitOnError = 0
        elif option in ('-p', '--prefix'):
            Prefix = argument
        elif option in ('-f', '--flatten'):
            Flatten = 1
        elif option in ('-r', '--raw', '--raw-errors'):
            RawErrors = 1
        elif option in ('-o', '--output'):
            Output = AbstractFile(argument, 'w', BufferedOutput)
        elif option in ('-a', '--append'):
            Output = AbstractFile(argument, 'a', BufferedOutput)
        elif option in ('-P', '--preprocess'):
            Preprocessing.append(('pre', argument))
        elif option in ('-I', '--import'):
            for module in string.split(argument, ','):
                module = string.strip(module)
                Preprocessing.append(('import', module))
        elif option in ('-E', '--execute'):
            Preprocessing.append(('exec', argument))
        elif option in ('-F', '--execute-file'):
            Preprocessing.append(('file', argument))
        elif option in ('-B', '--buffered-output'):
            BufferedOutput = 1
    if not remainder:
        remainder.append('-')
    # Set up the interpreter.
    if BufferedOutput and Output is None:
        raise CommandLineError, \
              "buffered output only makes sense with -o or -a arguments."
    interpreter = Interpreter(Output, Globals, Prefix, Flatten)
    try:
        # Execute command-line statements.
        i = 0
        for which, thing in Preprocessing:
            if which == 'pre':
                file = open(thing, 'r')
                interpreter.new(thing, file)
            elif which == 'exec':
                inputFile = StringIO.StringIO('@{%s}@\n' % thing)
                interpreter.new('<exec:%d>' % i, inputFile)
            elif which == 'file':
                thing = string.replace(thing, '"', '\\"')
                inputFile = StringIO.StringIO('@{execfile("%s")}@\n' % thing)
                interpreter.new('<file:%d (%s)>' % (i, thing), inputFile)
            elif which == 'import':
                inputFile = StringIO.StringIO('@{import %s}@\n' % thing)
                interpreter.new('<import:%d>' % i, inputFile)
            else:
                assert 0
            interpreter.wrap(BufferedOutput, RawErrors, ExitOnError)
            i = i + 1
        # Start processing files.
        for filename in remainder:
            if filename == '-':
                name = '<stdin>'
                file = sys.stdin
            else:
                name = filename
                file = open(filename, 'r')
            interpreter.new(name, file)
            interpreter.wrap(BufferedOutput, RawErrors, ExitOnError)
    finally:
        interpreter.finish()

def main():
    invoke(sys.argv[1:])

if __name__ == '__main__': main()
