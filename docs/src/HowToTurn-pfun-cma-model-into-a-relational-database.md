# How to turn pfun-cma-model into a relational database

## Use the VertexAI Workbench Instance:

https://4e3eb35e975d718d-dot-us-central1.notebooks.googleusercontent.com/lab/tree/notebook_template.ipynb

## Run the parameter grid script, store results in BigTable.

+ Use Feast to read/write to BigTable:
  + https://docs.feast.dev/
+ Results should include qualitative descriptions

## Rewrite the fitting procedure to perform a simple minimization between the transformed data & rows in the database.

+ Needs to be super fast & scalable.
+ The fastest performance option would be **BigTable** (NoSQL, super fast, high throughput).
