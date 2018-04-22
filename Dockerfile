FROM python

WORKDIR ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# esto es un try
MV ['./files/quiniela_as_results.csv', './old_resulsts/quiniela_as_results.csv']

WORKDIR ./quiniela_crawler

# lanzamos la spyder.
CMD ["scrapy crawl quiniela -o ../files/quiniela_as_results.csv -t csv"]
