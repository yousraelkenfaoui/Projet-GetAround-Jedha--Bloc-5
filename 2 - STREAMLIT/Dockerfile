# Use a base Python image
FROM python:3.9-slim

# Install the necessary dependencies
RUN pip install --upgrade pip
RUN pip install streamlit openpyxl matplotlib pandas numpy plotly seaborn

# Copy the source code into the container
COPY . /home/app

# Set the working directory
WORKDIR /home/app

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to start the Streamlit application
CMD ["streamlit", "run", "app.py"]
