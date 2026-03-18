#!/usr/bin/env sh

# scripts/convert-readme2html.sh

pandoc --from markdown \
       --to html \
       README.md | tee pfun_cma_model/static/README.html
