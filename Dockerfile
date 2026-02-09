# Use Python 3.11 slim image (TensorFlow compatible)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Set environment variable for Django
ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=server.settings

# Expose the port Render will use
EXPOSE 10000

# Run migrations and start Gunicorn
CMD ["bash", "-c", "python manage.py migrate && gunicorn server.wsgi:application --bind 0.0.0.0:$PORT"]