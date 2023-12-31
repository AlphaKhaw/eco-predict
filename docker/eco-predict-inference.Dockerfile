FROM python:3-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y

# Set working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only the necessary files
COPY ./models/weights/RANDOM_FOREST_20231027_195230_determined_edison.pkl /app/models/weights/RANDOM_FOREST_20231027_195230_determined_edison.pkl
COPY ./src/fastapi/main.py /app/src/fastapi/main.py
COPY ./src/inference/inference.py /app/src/inference/inference.py
COPY ./conf/base/inference.yaml /app/conf/base/inference.yaml
COPY ./src/model/model.py /app/src/model/model.py
COPY ./src/base/base_model.py /app/src/base/base_model.py
COPY ./src/enums/enums.py /app/src/enums/enums.py
COPY ./src/utils/eda/eda.py /app/src/utils/eda/eda.py

# Expose the port
EXPOSE 8000

# Use CMD to run the FastAPI application
CMD ["uvicorn", "src.fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
