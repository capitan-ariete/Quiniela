## To build dockerfile use command: 'docker build -t predictin .'

FROM python

WORKDIR ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
