

vis: topic.py .tweets.spacy
	knitr-spin-py topic.py
	Rscript -e "rmarkdown::render('topic.Rmd', output_dir='visualization')"
	rm topic.Rmd

script: topic.py .tweets.spacy
	python3 topic.py

.tweets.spacy: parse.py election-day-tweets/election-day-tweets.csv
	python3 parse.py


election-day-tweets/election-day-tweets.csv:
