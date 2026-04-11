# Use the official Go image as the base image
FROM golang:1.23-alpine AS builder

# Set the working directory
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o ollama-metrics-proxy .

# Use a minimal alpine image for the final stage
FROM alpine:latest

# Create a non-root user
RUN adduser -D -u 1000 appuser

# Set the working directory
WORKDIR /app

# Copy the binary from the builder stage
COPY --from=builder /app/ollama-metrics-proxy .

# Change ownership to the non-root user
RUN chown appuser:appuser /app/ollama-metrics-proxy

# Switch to the non-root user
USER appuser

# Expose the port (adjust as needed)
EXPOSE 8080

# Command to run the application
CMD ["./ollama-metrics-proxy"]