#!/bin/sh
find . -iname ".svn" -a -prune -o -type d -a -print | while read dir; do
    svn propget svn:ignore $dir 2>/dev/null | while read file; do
        test -z "$file" || rm -rf $dir/$file
    done
done
