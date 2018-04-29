## To build dockerfile use command: 'docker build -t spider .'

FROM python

WORKDIR ./

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# run spider.
CMD ["scrapy crawl quiniela -o ../files/file.csv -t csv"]

