WrapIDispatchMethod = function(obj, proto) {
    var m = proto.match(/^([\$a-zA-Z0-9_]+)\s*\(([^)]+)\)/);
    if (!m)
        return undefined;
    var nargs = m[2].split(/,/).length;

    var fun_body = 'var retval = function() { return obj.' + m[1] + '(';
    for (var i = 0; i < nargs; i++) {
        if (i > 0)
            fun_body += ', ';
        fun_body += 'arguments[' + i +']';
    }
    fun_body += '); };';
    eval(fun_body);
    return retval;
};

Object.mix = function(a, b) {
    var retval = {};

    for (var i in a) {
        retval[i] = a[i];
    }

    for (var i in b) {
        retval[i] = b[i];
    }

    return retval;
};

Object.map = function(obj, f) {
    var retval = {};

    for (var i in obj) {
        retval[i] = f(obj[i]);
    }

    return retval;
};

Object.prototype.mix = function(that) {
    return Object.mix(this, that);
};

Object.prototype.map = function(f) {
    return Object.map(this, f);
};

var BuildHelper = function () {
    this.initialize.apply(this, arguments);
};

BuildHelper.prototype = {
    initialize: function(buildInfo) {
        this.env = buildInfo.env;
        this.productInfo = buildInfo.productInfo;
        this.debugBuild = buildInfo.debugBuild;
        this.vcOutDir = FileSystemObject.GetAbsolutePathName(
                buildInfo.vcOutDir) + '\\';
        this.projectDir = FileSystemObject.GetAbsolutePathName(
                buildInfo.projectDir) + '\\';
        this.distDir = FileSystemObject.GetAbsolutePathName(
                buildInfo.distDir) + '\\';
        this.pythonHome = FileSystemObject.GetAbsolutePathName(
                buildInfo.pythonHome) + '\\';
        this.distIncludeDir = this.distDir + 'include\\';
        this.distIncludeEcellDir =
                this.distIncludeDir
                + this.productInfo.shortName + '-'
                + this.productInfo.version.major + '.'
                + this.productInfo.version.minor
                + '\\';
        this.distLibDir = this.distDir + 'lib\\';
        this.distLibEcellDir = this.distLibDir
                + this.productInfo.shortName + '-'
                + this.productInfo.version.major + '.'
                + this.productInfo.version.minor
                + '\\';
        this.distBinDir = this.distDir + 'bin\\';
        this.distDocDir = this.distDir + 'doc\\';
    },

    execPythonScript: function(args) {
        BuildHelper.exec(this.pythonHome + 'python', args);
    },

    run: function(tasks, method) {
        try {
            tasks.mix(this)[method]();
        } catch (e) {
            if (e instanceof String) {
                WScript.Echo("Error: " + e);
            } else if (e instanceof Error) {
                WScript.Echo("Error: " + e.description);
            }
            return 1;
        }

        return 0;
    }
};

BuildHelper.replacePlaceholders = function(str, vars) {
    return str.replace(/@([a-zA-Z0-9_-]+)@/g, function($0, $1) {
        return vars[$1];
    });
};

BuildHelper.processTemplate = function(outpath, inpath, vars) {
    var is = FileSystemObject.OpenTextFile(inpath, ForReading, false);
    var os = FileSystemObject.OpenTextFile(outpath, ForWriting, true);

    WScript.Echo('Generating ' + outpath + ' from ' + inpath);
    while (!is.AtEndOfStream) {
        var l = is.ReadLine();
        l = BuildHelper.replacePlaceholders(l, vars);
        os.WriteLine(l);
    }
    is.Close();
    os.Close();
};

BuildHelper.mkdir = function(path) {
    if (FileSystemObject.FolderExists(path))
        return true;

    var parentDir = FileSystemObject.GetParentFolderName(path);

    if (parentDir != '')
        this.mkdir(parentDir);

    WScript.Echo('Creating directory ' + path);
    FileSystemObject.CreateFolder(path);

    return true;
};

BuildHelper.copy = function(src, dest) {
    WScript.Echo('Copying ' + src + ' => ' + dest);
    FileSystemObject.CopyFile(src, dest);

    return true;
};

BuildHelper.copyMultiple = function(src_dir, dest, files) {
    var error = false;

    for (var i = 0; i < files.length; i++) {
        var src = src_dir + '\\' + files[i];
        try {
            this.copy(src, dest);
        } catch (e) {
            if (e instanceof Error) {
                WScript.Echo("Error: " + e.description);
                error = true;
            }
        }
    }

    if (error)
        throw new Error("One or more errors occurred in copyMultiple()");

    return true;
};

BuildHelper.chdir = function(dir) {
    WshShell.CurrentDirectory = dir;

    return true;
};

BuildHelper.escapeCommandlineArgument = function(arg) {
    return /\s/.test(arg) ?  '"' + arg.replace(/"/, '""') + '"': arg;
};

BuildHelper.exec = function(prog, args) {
    var cmdline = BuildHelper.escapeCommandlineArgument(prog);

    for (var i = 0; i < args.length; i++) {
        cmdline += ' ' + BuildHelper.escapeCommandlineArgument(args[i]);
    }

    var ex = WshShell.Exec(cmdline);
    while (ex.Status == 0) {
        WScript.Echo(cmdline);
        if (!ex.StdOut.AtEndOfStream)
            WScript.Echo(ex.StdOut.ReadAll());
        if (!ex.StdErr.AtEndOfStream)
            WScript.Echo(ex.StdErr.ReadAll());
        WScript.Sleep(100);
    }

    if (ex.Status != 1)
        throw "Failed to execute " + prog;
};

BuildHelper.ArgsParser = function() {
    this.initialize.apply(this, arguments);
};

BuildHelper.ArgsParser.StandardBuilder = function() {
    this.initialize.apply(this, arguments);
};

BuildHelper.ArgsParser.StandardBuilder.prototype = {
    initialize: function() {
        this.real_args = [];
        this.options = {};
    },

    receiveRealArgument: function(arg) {
        this.real_args.push(arg);
    },

    receiveOption: function(opt, val) {
        if (this.options[opt.id] === undefined) {
            this.options[opt.id] = val;
        } else {
            if (typeof(this.options[opt.id]) == 'string')
                this.options[opt.id] = [ this.options[opt.id] ];
            this.options[opt.id].push(val);
        }
    }
};

BuildHelper.ArgsParser.prototype = {
    initialize: function(opts) {
        var long_opt_map = {};
        var short_opt_list = [];

        for (var i = 0; i < opts.length; i++) {
            if (opts[i].long !== undefined)
                long_opt_map[opts[i].long] = opts[i];
            if (opts[i].short !== undefined) {
                short_opt_list.push(opts[i]);
            }
        }

        this.long_opt_map = long_opt_map;
        this.short_opt_list = short_opt_list;
    },

    parse: function(argv) {
        var state = 0;
        var handler = null;
        if (arguments.length > 1)
            handler = arguments[1];
        else
            handler = new BuildHelper.ArgsParser.StandardBuilder();

        var opt = null;
        for (var i = 0; i < argv.length; i++) {
            var arg = argv.Item(i);

            switch (state) {
            case 0:
                if (arg.charAt(0) == '-') {
                    var opt_value = null;

                    if (arg.charAt(1) == '-') {
                        var tmp = arg.match(/^--([^=]+)(?:=(.*))?/, arg);
                        opt_name = tmp[1];
                        if (this.long_opt_map[opt_name] === undefined)
                            return 'Unknown option --' + opt_name;
                        opt = long_opt_map[opt_name];
                        if (tmp[2] !== undefined) {
                            switch (opt.operand) {
                            case 0:
                                return 'Option --' + opt_name + ' takes no operand';
                            case 1:
                                return 'Invalid operand for option --' + opt_name;
                            }
                            opt_value = tmp[2];
                        }
                    } else {
                        var opt_name = arg.substring(1, arg.length);

                        for (var i = this.short_opt_list.length; --i >= 0;) {
                            if (opt_name.indexOf(this.short_opt_list[i].short) == 0) {
                                opt = this.short_opt_list[i];
                                break;
                            }
                        }

                        if (!opt)
                            return 'Unknown option -' + opt_name.charAt(0);

                        switch (opt.operand) {
                        case 3:
                            opt_value = opt_name.substring(1);
                            break;
                        case 2:
                            if (opt_name.length == opt.short.length)
                                return 'Missing operand for option -' + opt.short;
                            if (opt_name.charAt(opt.short.length) != '=')
                                return 'Invalid operand for option -' + opt.short;
                            opt_value = opt_name.substring(2, opt_name.length);
                            break;
                        default:
                            if (opt_name.length != opt.short.length)
                                return 'Invalid operand for option -' + opt.short;
                        }
                    }
                }

                if (!opt) {
                    if (handler.receiveRealArgument)
                        handler.receiveRealArgument(arg);
                } else {
                    if (opt.operand == 1) {
                        state = 1;
                    } else {
                        if (opt.receiver)
                            opt.receiver.call(handler, opt, opt_value);
                        else
                            handler.receiveOption(opt, opt_value);
                        opt = null;
                    }
                }
                break;
            case 1:
                if (opt.receiver)
                    opt.receiver.call(handler, opt, arg);
                else
                    handler.receiveOption(opt, arg);
                opt = null;
                state = 0;
            }
        }

        if (state == 1) {
            return 'Missing operand for option ' + arg;
        }

        return handler;
    }
};

BuildHelper.BourneShellNotationReader = function() {
    this.initialize.apply(this, arguments);
};

BuildHelper.BourneShellNotationReader.prototype = {
    initialize: function() {
        this.env = {};
    },

    expand: function(txt) {
        var self = this;
        return txt.replace(/\$(?:([a-zA-Z0-9_]+)|(\{[^}]+\}))/g,
                function($0, $1, $2) {
                    return $2 ? self.env[$2.substring(1, $2.length - 1)]:
                            self.env[$1];
                });
    },

    eval: function(obj) {
        var pat1 = /^\s*([^=]*)=([^'"][^ ]*|"(?:[^"]|\\")+("\s*)?|'[^']*('\s*))?$/;
        var pat2 = /^(?:[^"]|\\")+("\s*)$/;
        var pat3 = /^[^']+('\s*)$/;

        if (typeof(obj) != 'string' && !(obj instanceof String)) {
            return this.eval(obj.OpenAsTextStream(ForReading).ReadAll());
        }

        var lines = obj.split(/\r\n|\n|\r/);
        var state = 0;
        var name, val;

        for (var i = 0, l = lines.length; i < l; i++) {
            var line = lines[i];
            switch (state) {
            case 0:
                var m = line.match(pat1);
                if (m) {
                    val = '';
                    switch (m[2].charAt(0)) {
                    case '"':
                        if (!m[3]) {
                            name = m[1];
                            state = 1;
                            val = m[2].substring(1, m[2].length) + "\n";
                            continue;
                        }
                        val = m[2].substring(1, m[2].length - 1);
                        break;
                    case "'":
                        if (!m[4]) {
                            name = m[1];
                            state = 2;
                            val = m[2].substring(1, m[2].length) + "\n";
                            continue;
                        }
                        val = m[2].substring(1, m[2].length - 1);
                        break;
                    default:
                        val = m[2];
                        break;
                    }
                    this.env[m[1]] = val;
                }
                break;
            case 1:
                var m = line.match(pat2);
                if (m) {
                    val += line.substr(0, line.length - m[1].length);
                    this.env[name] = this.expand(val);
                    state = 0;
                } else {
                    val += line + "\n";
                }
                break; 
            case 2:
                var m = line.match(pat3);
                if (m) {
                    val += line.substr(0, line.length - m[2].length);
                    this.env[name] = val;
                    state = 0;
                } else {
                    val += line + "\n";
                }
                break; 
            }
        }
    }
};
