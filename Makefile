# Root Makefile

# List of subdirectories to traverse
SUBDIRS = runtime pfun-cma-model-reheater

.PHONY: all $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) --jobs=4 -C $@

# Clean target to remove intermediate files
clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
