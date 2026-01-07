# -------------------------------
# Dockerfile for Diabetes Prediction API
# -------------------------------

# Use official lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first for caching layer
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# -------------------------------
# Command to run FastAPI app with Uvicorn
# Adjust module path to Web_Service or Web_Service_batch
# -------------------------------

# For single prediction API
# CMD ["uvicorn", "Train_models.Web_Service:app", "--host", "0.0.0.0", "--port", "8000"]

# For batch prediction API
CMD ["uvicorn", "Train_models.Web_Service_batch:app", "--host", "0.0.0.0", "--port", "8000"]
