import scrapy

"""
Crawl results URLs such as
https://resultados.as.com/resultados/futbol/primera/2017_2018/jornada/regular_a_30/
and extract match and result.
"""


class BlogSpider(scrapy.Spider):
    name = 'quiniela'

    def start_requests(self):
        urls = []
        jornadas = 38

        for i in range(1, jornadas+1):
            urls += ['https://resultados.as.com/resultados/futbol/primera/'
                     '2017_2018/jornada/regular_a_{}/'.format(i)]

        for url in urls:
            try:
                yield scrapy.Request(url=url, callback=self.parse)
            except ValueError:
                logger.warning('URL {} does not exist'.format(url))

    def parse(self, response):

        for match in response.xpath("//div[@class='cont-resultado finalizado']"):

            yield {'jornada': response.url,
                   'match': match.xpath("./a/@title").extract_first().strip(),
                   'result': match.xpath("./a/text()").extract_first().strip()}
