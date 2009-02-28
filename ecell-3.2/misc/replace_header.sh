#!/bin/sh

if [ $# = 0 ]; then
    echo "usage: `basename $0` template file [file] ..." >&2
    exit 255
fi

TEMPLATE_FILE=$1
shift

while [ ! -z $1 ]; do
    if grep "END_HEADER" $1 > /dev/null; then
        sed -i -e "
/END_HEADER/ {
    r $TEMPLATE_FILE
}
1,/END_HEADER/ d
    " "$1"
    else
        cat "$TEMPLATE_FILE" "$1" > "$1.tmp"
        cp "$1.tmp" "$1"
        rm "$1.tmp"
    fi
    shift
done
