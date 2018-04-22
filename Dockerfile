FROM python

WORKDIR ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Esto debería estar en un try!!
MV ['./files/quiniela_as_results.csv', './old_resulsts/quiniela_as_results.csv']

WORKDIR ./quiniela_crawler

# run spider.
CMD ["scrapy crawl quiniela -o ../files/quiniela_as_results.csv -t csv"]

# run features generator
WORKDIR ../

CMD ["python", "features_generator.py"]

# run predictor
CMD ["python", "quiniela_predictor.py"]