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

    @staticmethod
    def _urls_generator():
        """
        Generate list of URLS to craw
        :return:
        """
        list_years = list(zip(range(2013, 2019), range(2014, 2020)))
        jornadas = range(1, 39)
        ligas = [
            'primera',
            'segunda',
        ]

        urls = [f'https://resultados.as.com/resultados/futbol/' \
                f'{liga}/' \
                f'{start_year}_{end_year}/' \
                f'jornada/regular_a_{jornada}/'
                for liga in ligas
                for jornada in jornadas
                for start_year, end_year in list_years]

        return urls

    def start_requests(self):
        """
        Crawl leagues seassons from 2013-2014 onwards.
        :return:
        """
        urls = self._urls_generator()

        for url in urls:
            try:
                yield scrapy.Request(url=url, callback=self.parse)
            except ValueError:
                logger.warning('URL {} does not exist'.format(url))

    def parse(self, response):
        """
        Parse the match result
        :param response:
        :return:
        """
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
