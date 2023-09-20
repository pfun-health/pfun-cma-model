# Root Makefile

# List of subdirectories to traverse
SUBDIRS ?= runtime pfun-cma-model-reheater

.PHONY: all $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) --jobs=4 -C $@

# Clean target to remove intermediate files
clean:
	sh -c 'set +e; rm -rf ./dist/*'
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
build:
	poetry build

install:
	poetry build
	poetry install

publish:
	poetry publish