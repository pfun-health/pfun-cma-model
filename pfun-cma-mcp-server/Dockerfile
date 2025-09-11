FROM node:10-alpine as node
# Set working directory
WORKDIR /app
# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies and build the project
FROM node as deps

RUN \
    npm install -g typescript && \
    npm update && \
    npm install
# Build the project
FROM node as builder
WORKDIR /app
COPY --from=deps /app .
RUN npm run build


# Final stage: setup the runtime environment
FROM builder as runner
WORKDIR /app
COPY --from=builder /app .
