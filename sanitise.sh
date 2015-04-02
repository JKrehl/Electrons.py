#!/bin/sh
shopt -s globstar
rm -v **/*.{py~,pyc}
