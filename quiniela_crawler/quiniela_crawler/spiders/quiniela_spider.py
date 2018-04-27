import scrapy

"""
Crawl results URLs such as
https://resultados.as.com/resultados/futbol/primera/2017_2018/jornada/regular_a_30/
https://resultados.as.com/resultados/futbol/segunda/2017_2018/jornada/regular_a_30/
and extract match and result.

To run it use:
cd quiniela_crawler
scrapy crawl quiniela -o ../files/files_new.csv
"""


class BlogSpider(scrapy.Spider):
    name = 'quiniela'

    def start_requests(self):
        jornadas = 38
        ligas = ['primera', 'segunda']

        urls = ['https://resultados.as.com/resultados/futbol/' \
                '{liga}/2017_2018/jornada/regular_a_{jornada}/'.format(jornada=j, liga=l)
                for l in ligas
                for j in range(1, jornadas + 1)]

        for url in urls:
            try:
                yield scrapy.Request(url=url, callback=self.parse)
            except ValueError:
                logger.warning('URL {} does not exist'.format(url))

    def parse(self, response):

        for match in response.xpath("//div[@class='cont-resultado finalizado']"):

            d = {'jornada': response.url,
                 'match': match.xpath("./a/@title").extract_first().strip(),
                 'result': match.xpath("./a/text()").extract_first().strip()}

            if 'primera' in response.url:
                d['liga'] = 'primera'
            elif 'segunda' in response.url:
                d['liga'] = 'segunda'
            else:
                d['liga'] = 'unknown'

            yield d
