# Use Python 3.11 slim (TensorFlow compatible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set environment variables for Django
ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=server.settings

# Expose the port that Render will use
EXPOSE 10000

# Run migrations and start Gunicorn
CMD ["bash", "-c", "python manage.py migrate && gunicorn server.wsgi:application --bind 0.0.0.0:$PORT"]
