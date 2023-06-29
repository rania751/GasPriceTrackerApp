# Use an official Node.js runtime as the base image
FROM node:14-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the package.json and package-lock.json files to the container
COPY package*.json /app/

# Install the dependencies
RUN npm install

# Copy the entire application code to the container
COPY . .

# Build the React application
RUN npm run build

# Expose the port that the application will run on
EXPOSE 3000

# Define the command to run the application
CMD ["npm", "start"]
