FROM python:3.11

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN pip install flask opencv-contrib-python==4.10.0.84 numpy

EXPOSE 5000

CMD ["python", "main2.py"]
