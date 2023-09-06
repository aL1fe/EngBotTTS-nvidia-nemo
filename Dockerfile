FROM nvcr.io/nvidia/nemo:23.06

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]
