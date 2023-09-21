# Use a lightweight base image
FROM python:3.10-alpine

# Install dependencies
COPY requirements*.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-dev.txt

# Copy the application
COPY . /app
WORKDIR /app

# Expose the port the app runs on
EXPOSE 8002

# Command to run the application
CMD ["chalice", "local"]
