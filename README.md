# Quiniela
Crawler and Data Science. Crawling www.as.com for "La Liga" results and build a quiniela predictor for spanish football league (La Liga).

<h3> 1. Crawl football results</h3>
Crawl this
<a href="https://resultados.as.com/resultados/futbol/primera/2017_2018/jornada/regular_a_*/">online resource</a>
using scrapy (spider <a href="using scrapy at Quiniela/quiniela_crawler/quiniela_crawler/spiders/quiniela_spider.py">here</a>)

<h3> 2. Create features dataset</h3>
Create features dataset from the data crawled from the internet.
(script <a href="Quiniela/features_generator.py">here</a>)

<h3> 3. [PENDING] Predict next 'jornada' </h3>
[PENDING] Predict next jornada results to play Quiniela.

<h3> Bibliography </h3>

I strongly recommend following read (it has nothing to do with my method -mine is far less sophisticated-)
<a href="https://arxiv.org/pdf/1710.02824.pdf">Paper Beating Bookies</a> 

