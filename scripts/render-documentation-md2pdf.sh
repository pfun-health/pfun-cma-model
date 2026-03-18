#!/usr/bin/env sh

# scripts/render-documentation-md2pdf.sh :
# A script to convert all Markdown documentation files in the current directory to PDF format using Pandoc.

# Go through all the Markdown files present in the current directory
for file in *.md; do
	# Convert each Markdown file to PDF
	echo "Converting $file to PDF..."
	pandoc "$file" -o "./docs/${file%.md}.pdf"
done

echo "All conversions complete!"
