#!/usr/local/bin/python
#
# $Id$ $Date$

"""
A system for processing Python as markup embedded in text.
"""

__program__ = 'em'
__version__ = '1.3'
__url__ = 'http://www.alcyone.com/pyos/em/'
__author__ = 'Erik Max Francis <max@alcyone.com>'
__copyright__ = 'Copyright (C) 2002 Erik Max Francis'
__license__ = 'GPL'


import getopt
import string
import cStringIO
import sys
import types

StringIO = cStringIO
del cStringIO


FAILURE_CODE = 1
DEFAULT_PREFIX = '@'
INTERNAL_MODULE_NAME = 'em'
SECONDARY_CHARS = "#%({[<` \t\v\n"
SIGNIFICATOR_RE_STRING = r"@%(\S)+\s*(.*)$"


class EmpyError(Exception):
    
    """The base class for all em errors."""

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


class ProxyFile:
    
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
        """Install a new filter.  Handle all the special types of filters
        here."""
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


class Parser:
    
    """The core parser which also manages the output file."""
    
    IDENTIFIER_FIRST = '_abcdefghijklmnopqrstuvwxyz' + \
                       'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    IDENTIFIER_REST = '0123456789' + IDENTIFIER_FIRST + '.'

    def __init__(self, interpreter, proxy, globals, \
                 locals=None):
        self.interpreter = interpreter
        self.proxy = proxy
        self.buffer = ''
        self.globals = globals
        self.locals = locals
        self.first = 1 # have we not yet started?

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
            elif end[0] in self.interpreter.prefix + SECONDARY_CHARS:
                code, end = end[0], end[1:]
                return start, self.interpreter.prefix + code, end
            elif end[0] in Parser.IDENTIFIER_FIRST:
                return start, self.interpreter.prefix, end
            else:
                raise ParseError, "unrecognized token sequence"

    def scanNext(self, data, target, i=0, j=None):
        """Scan from i to j for the next occurrence of one of the characters
        in the target string, respecting string literals."""
        quote = None
        if j is None:
            j = len(data)
        while i < j:
            c = data[i]
            if c == "'" or c == '"':
                if not quote:
                    quote = c
                elif quote == c:
                    quote = None
            elif quote:
                if c == '\\':
                    i = i + 1
            elif not quote:
                if c in target:
                    return i
            i = i + 1
        else:
            raise TransientParseError, "expecting ending character"

    def scanNextMandatory(self, data, target, i, j):
        """Scan from i to j for the next occurrence of one of the characters
        in the target string, respecting string literals.  If the character
        is not present, it is a parse error."""
        quote = None
        while i < j:
            c = data[i]
            if c == "'" or c == '"':
                if not quote:
                    quote = c
                elif quote == c:
                    quote = None
            elif quote:
                if c == '\\':
                    i = i + 1
            elif not quote:
                if c in target:
                    return i
            i = i + 1
        else:
            raise ParseError, "expecting %s, not found" % target

    def scanComplex(self, data, enter, exit, i=0, j=None):
        """Scan from i for an ending sequence, respecting entries and exits,
        respecting string literals."""
        quote = None
        depth = 0
        if j is None:
            j = len(data)
        while i < j:
            c = data[i]
            if c == "'" or c == '"':
                if not quote:
                    quote = c
                elif quote == c:
                    quote = None
            elif quote:
                if c == '\\':
                    i = i + 1
            elif not quote:
                if c == enter:
                    depth = depth + 1
                elif c == exit:
                    depth = depth - 1
                    if depth < 0:
                        return i
            i = i + 1
        else:
            raise TransientParseError, "expecting end of complex expression"

    def scanWord(self, data, i=0):
        """Scan from i for a simple word."""
        dataLen = len(data)
        while i < dataLen:
            if not data[i] in Parser.IDENTIFIER_REST:
                return i
            i = i + 1
        else:
            raise TransientParseError, "expecting end of word"

    def scanPhrase(self, data, i=0):
        """Scan from i for a phrase (e.g., 'word', 'f(a, b, c)', 'a[i]', or
        combinations like 'x[i](a)'."""
        # Find the word.
        i = self.scanWord(data, i)
        while i < len(data) and data[i] in '([':
            enter = data[i]
            exit = {'(': ')', '[': ']',}[enter]
            i = self.scanComplex(data, enter, exit, i + 1) + 1
        return i
    
    def scanSimple(self, data, i=0):
        """Scan from i for a simple expression, which consists of one 
        more phrases separated by dots."""
        i = self.scanPhrase(data, i)
        dataLen = len(data)
        while i < dataLen and data[i] == '.':
            i = self.scanPhrase(data, i)
        # Make sure we don't end with a trailing dot.
        while i > 0 and data[i - 1] == '.':
            i = i - 1
        return i

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
            self.proxy.write(start)
            if token is None:
                # No remaining tokens, just pass through.
                self.proxy.write(self.buffer)
                self.buffer = ''
                return 0
            elif token in (prefix + ' ', prefix + '\t', \
                           prefix + '\v', prefix + '\n'):
                # Whitespace/line continuation.
                pass
            elif token == prefix:
                # "Simple expression" expansion.
                try:
                    i = self.scanSimple(self.buffer)
                    code, self.buffer = self.buffer[:i], self.buffer[i:]
                    result = self.evaluate(code)
                    if result is not None:
                        self.proxy.write(str(result))
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
                # Expression evaluation.
                try:
                    i = self.scanComplex(self.buffer, '(', ')', 0)
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    result = self.evaluate(code)
                    if result is not None:
                        self.proxy.write(str(result))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '`':
                # Repr evalulation.
                try:
                    i = self.scanNext(self.buffer, '`', 0)
                    code, self.buffer = self.buffer[:i], self.buffer[i + 1:]
                    self.proxy.write(repr(self.evaluate(code)))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '[':
                # Conditional evaluation.
                try:
                    k = self.scanComplex(self.buffer, '[', ']', 0)
                    i = self.scanNextMandatory(self.buffer, '?', 0, k)
                    try:
                        j = self.scanNextMandatory(self.buffer, ':', i, k)
                    except ParseError:
                        j = k
                    testCode = self.buffer[:i]
                    thenCode = self.buffer[i + 1:j]
                    elseCode = self.buffer[j + 1:k]
                    self.buffer = self.buffer[k + 1:]
                    result = self.evaluate(testCode)
                    if result:
                        expansion = self.evaluate(thenCode)
                    else:
                        if elseCode:
                            expansion = self.evaluate(elseCode)
                        else:
                            expansion = None
                    if expansion is not None:
                        self.proxy.write(str(expansion))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '<':
                # Protected evaluation.
                try:
                    j = self.scanComplex(self.buffer, '<', '>', 0)
                    try:
                        i = self.scanNextMandatory(self.buffer, ':', 0, j)
                    except ParseError:
                        i = j
                    tryCode = self.buffer[:i]
                    catchCode = self.buffer[i + 1:j]
                    self.buffer = self.buffer[j + 1:]
                    try:
                        expansion = self.evaluate(tryCode)
                    except SyntaxError:
                        raise # let syntax errors through
                    except:
                        if catchCode:
                            expansion = self.evaluate(catchCode)
                        else:
                            expansion = None
                    if expansion is not None:
                        self.proxy.write(str(expansion))
                except TransientParseError:
                    self.buffer = token + self.buffer
                    raise
            elif token == prefix + '{':
                # Statement evaluation.
                try:
                    i = self.scanComplex(self.buffer, '{', '}', 0)
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
                self.proxy.write(prefix)
            else:
                raise ParseError, "unknown token: %s" % token
            return 1
        else:
            return 0

    def parse(self):
        while self.buffer:
            self.once()

    def done(self):
        """Declare that this parsing session is over and that any unparsed
        data should be considered an error."""
        if self.buffer:
            if self.buffer[-1] != '\n':
                # The data is not well-formed text; add a newline.
                self.buffer = self.buffer + '\n'
            # A TransientParseError thrown here is a real parse error.
            self.parse()
            self.buffer = ''

    def evaluate(self, expression):
        """Evaluate an expression."""
        if self.locals:
            return eval(expression, self.globals, self.locals)
        else:
            return eval(expression, self.globals)

    def execute(self, statements):
        """Execute a statement."""
        if string.find(statements, '\n') < 0:
            statements = string.strip(statements)
        if self.locals:
            exec statements in self.globals, self.locals
        else:
            exec statements in self.globals

    def significate(self, key, value):
        """Declare a significator."""
        name = '__%s__' % key
        self.globals[name] = value

    def __nonzero__(self): return len(self.buffer) != 0


class Context:
    
    """An interpreter context, which encapsulates a name, an input
    file object, and a parser object."""

    def __init__(self, name, input, parser, line=0):
        self.name = name
        self.input = input
        self.parser = parser
        self.line = line
        self.exhausted = 1


class Stack:
    
    """A simple stack that behave as a sequence (with 0 being the top
    of the stack, not the bottom)."""

    def __init__(self, seq=None):
        if seq is None:
            seq = []
        self.data = seq

    def top(self):
        try:
            return self.data[-1]
        except IndexError:
            raise StackUnderflowError, "stack is empty for top"
        
    def pop(self):
        try:
            return self.data.pop()
        except IndexError:
            raise StackUnderflowError, "stack is empty for pop"
        
    def push(self, object): self.data.append(object)

    def clone(self):
        """Create a duplicate of this stack."""
        return self.__class__(self.data[:])

    def __nonzero__(self): return len(self.data) != 0
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[-(index + 1)]


class PseudoModule:
    
    """A pseudomodule for the builtin em routines."""

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
        file = StringIO.StringIO(data)
        self.interpreter.new('<expand>', file, locals)

    def quote(self, data):
        """Quote the given string so that if it were expanded it would
        evaluate to the original."""
        result = ''
        quote = None
        for c in data:
            if c == "'" or c == '"':
                if not quote:
                    quote = c
                elif quote == c:
                    quote = None
            if c == self.interpreter.prefix and not quote:
                result = result + c * 2
            else:
                result = result + c
        return result

    # Pseudomodule manipulation.

    def flatten(self):
        """Flatten the contents of the pseudo-module into the globals
        namespace."""
        dict = {}
        dict.update(self.__dict__)
        # The pseudomodule is really a class instance, so we need to
        # fumble through the parent class' __dict__ as well, but to
        # get proper bound methods, we need to call getattr on self.
        for attribute in self.__class__.__dict__.keys():
            dict[attribute] = getattr(self, attribute)
        # Stomp everything into the globals namespace.
        for key, value in dict.items():
            self.interpreter.globals[key] = value

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
        self.interpreter.proxy.revert()

    def startDiversion(self, diversion):
        """Start diverting to the given diversion name."""
        self.interpreter.proxy.divert(diversion)

    def playDiversion(self, diversion):
        """Play the given diversion and then purge it."""
        self.interpreter.proxy.undivert(diversion, 1)

    def replayDiversion(self, diversion):
        """Replay the diversion without purging it."""
        self.interpreter.proxy.undivert(diversion, 0)

    def purgeDiversion(self, diversion):
        """Eliminate the given diversion."""
        self.interpreter.proxy.purge(diversion)

    def playAllDiversions(self):
        """Play all existing diversions and then purge them."""
        self.interpreter.proxy.undivertAll(1)

    def replayAllDiversions(self):
        """Replay all existing diversions without purging them."""
        self.interpreter.proxy.undivertAll(0)

    def purgeAllDiversions(self):
        """Purge all existing diversions."""
        self.interpreter.proxy.purgeAll()

    def getCurrentDiversion(self):
        """Get the name of the current diversion."""
        return self.interpreter.proxy.currentDiversion

    def getAllDiversions(self):
        """Get the names of all existing diversions."""
        diversions = self.interpreter.proxy.diversions.keys()
        diversions.sort()
        return diversions
    
    # Filter.

    def resetFilter(self):
        """Reset the filter so that it does no filtering."""
        self.interpreter.proxy.install(None)

    def nullFilter(self):
        """Install a filter that will consume all text."""
        self.interpreter.proxy.install(0)

    def getFilter(self):
        """Get the current filter."""
        filter = self.interpreter.proxy.filter
        if filter is self.interpreter.proxy.file:
            return None
        else:
            return filter

    def setFilter(self, filter):
        """Set the filter."""
        self.interpreter.proxy.install(filter)


class Interpreter:
    
    """Higher-level manipulation of a stack of parsers, with file/line
    context."""

    def __init__(self, output=None, globals=None, \
                 prefix=DEFAULT_PREFIX, flatten=0):
        # Set up the proxy first, since .pseudo needs it.
        if output is None:
            output = sys.stdout
        self.output = output
        self.proxy = ProxyFile(output)
        if globals is None:
            pseudo = PseudoModule(self)
            globals = {pseudo.__name__: pseudo}
        self.globals = globals
        self.prefix = prefix
        self.contexts = Stack()
        if flatten:
            pseudo.flatten()
        # Override sys.stdout with the proxy file object.  This could
        # potentially be done better.
        sys.stdout = self.proxy

    def __del__(self):
        self.finish()

    def new(self, name, input, locals=None):
        """Create a new interpreter context (a new file object to process)."""
        parser = Parser(self, self.proxy, self.globals, locals)
        context = Context(name, input, parser)
        self.contexts.push(context)

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
        """Declare this interpreting session over; close the proxy file
        object."""
        if self.proxy is not None:
            self.proxy.close()
            self.proxy = None

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
  -P --preprocess=<filename>   Interpret em file before main processing
  -I --import=<modules>        Import Python modules before processing
  -E --execute=<statement>     Execute Python statement before processing
  -F --execute-file=<filename> Execute Python file before processing
  -B --buffered-output         Fully buffer output (even open) with -o or -a

The following expansions are supported (where @ is the prefix):
  @# comment nl                Comment; remove everything up to newline
  @ whitespace                 Remove following whitespace; line continuation
  @@                           Literal @; @ is escaped (duplicated prefix)
  @( expression )              Evaluate expression and substitute with str
  @ simple_expression          Evaluate simple expression and substitute;
                               e.g., @x, @x.y, @f(a, b), @l[i], etc.
  @` expression `              Evaluate expression and substitute with repr
  @[ if ? then : else ]        Conditional expression evaluation
  @< try : catch >             Protected expression evaluation
  @{ statements }              Statements are executed for side effects
  @%% key whitespace value nl   Significator form of __key__ = value

The %s pseudomodule contains the following attributes:
  VERSION                      String representing em version
  SIGNIFICATOR_RE_STRING       Regular expression string matching significators
  interpreter                  Currently-executing interpreter instance
  identify()                   Identify top context as name, line
  Filter                       The base class for custom filters
  include(file, [locals])      Include filename or file-like object
  expand(string, [locals])     Explicitly expand string and return expansion
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
    """Run a standalone instance of an em interpeter."""
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
