FROM python

WORKDIR ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR ./quiniela_crawler

CMD ["scrapy crawl quiniela"]
