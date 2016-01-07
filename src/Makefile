# This is the top-level Makefile for this package

SUBDIRS = ivector ivectorbin

all : subdirs
.PHONY : all 

subdirs: 
	for dir in $(SUBDIRS); do echo $$dir; $(MAKE) -C $$dir; done
