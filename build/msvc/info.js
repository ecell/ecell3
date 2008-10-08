buildInfo.env = (function() {
    var sh = new BuildHelper.BourneShellNotationReader();
    WScript.Echo(buildInfo.projectDir + '\\ecell_version.sh');
    sh.eval(FileSystemObject.GetFile(
            buildInfo.projectDir + '\\ecell_version.sh'));
    
    return sh.env.mix({
        top_srcdir: buildInfo.projectDir + '\\ecell',
        ECELL_VERSION_STRING: '"' + sh.env.ECELL_VERSION_NUMBER + '"',
        VERSION: sh.env.ECELL_VERSION_NUMBER,
        INCLTDL: '/I' + buildInfo.projectDir + '\\libltdl',
        DMTOOL_INCLUDE_DIR: buildInfo.projectDir,
        NUMPY_INCLUDE_DIR: buildInfo.pythonHome + '\\lib\\site-packages\\numpy\\core\\include'
    });
})();

buildInfo.productInfo = {
	name: 'E-Cell',
	shortName: 'ecell',
    version: {
        major: buildInfo.env.ECELL_MAJOR_VERSION,
        minor: buildInfo.env.ECELL_MINOR_VERSION,
        micro: buildInfo.env.ECELL_MICRO_VERSION
    }
};
