# -------------------------------
# 1. Use official Python image
# -------------------------------
FROM python:3.10.11-slim

# -------------------------------
# 2. Set working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 3. Copy requirements first (for caching)
# -------------------------------
COPY requirements.txt .

# -------------------------------
# 4. Install dependencies
# -------------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 5. Copy the rest of the application
# -------------------------------
COPY . .

# -------------------------------
# 6. Expose the port FastAPI runs on
# -------------------------------
EXPOSE 8000

# -------------------------------
# 7. Command to run the app with uvicorn
# -------------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
