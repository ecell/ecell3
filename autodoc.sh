#!/bin/sh

WORKDIR=/tmp/autodoc.$$

TARGETDIR=.

CVS="cvs -z3"
unset CVS_RSH

export CVSROOT=":pserver:anonymous@bioinformatics.org:/cvsroot"
PACKAGE=ecell3



initialize()
{
        trap interrupted 2 9 15
        run_command mkdir $WORKDIR
}

cleanup()
{
        run_command rm -rf $WORKDIR
}

interrupted()
{
        echo "interrupted..."
        trap '' 2 9 15
        cleanup
        exit 1
}


run_command()
{
        if [ $TEST_MODE ] ; then
                echo $@
        else
                [ $VERBOSE ] && echo $@
                $@
        fi
return $?
}



exit_if_failed()
{
    $@
    if [ $? -ne 0 ] ; then
	echo "error exit.."
	cleanup
        exit $?
    fi
}

cvs_login()
{
    expect -f - <<ENDCAT
    spawn $CVS login

    expect assword:
    send   \n
ENDCAT
}


initialize

cd $WORKDIR

echo 'logging in '

exit_if_failed cvs_login

exit_if_failed $CVS checkout $PACKAGE

cd $PACKAGE

exit_if_failed ./autogen.sh
exit_if_failed make doc
exit_if_failed make doc-ps
exit_if_failed make doc-pdf
exit_if_failed make doc-html-archive


exit_if_failed cp -a doc/refman $TARGETDIR
#exit_if_failed cp -a doc/??? $TARGET

cleanup
