# Use Node LTS
FROM node:20-alpine

# Set working directory
WORKDIR /usr/src/app

# Copy package files first (better cache)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy rest of the frontend code
COPY . .

# Expose Next.js port
EXPOSE 3000

# Start Next.js in dev mode
CMD ["npm", "run", "dev"]
