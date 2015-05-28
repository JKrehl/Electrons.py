#!/bin/sh
shopt -s globstar
rm -v **/*.{py~,pyc,html,c,cpp,pyxbldc,pyx~}
