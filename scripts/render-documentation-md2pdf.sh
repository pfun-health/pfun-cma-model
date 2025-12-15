#!/usr/bin/env bash

# Go through all the Markdown files present in the current directory
for file in *.md; do
  # Convert each Markdown file to PDF
  pandoc "$file" -o "./docs/${file%.md}.pdf"
done

echo "All conversions complete!"
