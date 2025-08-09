# Use the official k6 image as a base
FROM grafana/k6:latest

# Copy all local k6 test scripts from the testing directory into the container
# The working directory is set to /home/k6 by the base image
COPY . /home/k6/
