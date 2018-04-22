import os
import logging.config
from logging.config import fileConfig
import scrapy

if not os.path.isdir('logs'):
    os.makedirs('logs')
if not os.path.isfile('logs/python.log'):
    os.mknod('logs/python.log')

logger = logging.getLogger(__name__)
fileConfig('logger.ini')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BlogSpider(scrapy.Spider):
    name = 'quiniela'
    jornadas = 38

    def start_requests(self):
        urls = []

        for i in range(1, jornadas+1):
            urls += ['https://resultados.as.com/resultados/futbol/primera/'
                     '2017_2018/jornada/regular_a_{}/'.format(i)]

        for url in urls:
            try:
                yield scrapy.Request(url=url, callback=self.parse)
            except ValueError:
                logger.warning('URL {} does not exist'.format(url))

    def parse(self, response):

        for next_match in response.css('div.cont-resultado finalizado'):
            yield {'match': next_match.css('a.title').extract_first(),
                   'results': next_match.css('a ::text').extract_first()}
