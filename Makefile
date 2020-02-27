

vis: model.py .tweets.spacy
	knitr-spin-py model.py
	Rscript -e "library(rmarkdown); rmarkdown::render('model.Rmd', output_dir='visualization')"
	rm model.Rmd


script: model.py .tweets.spacy
	python3 model.py

.tweets.spacy: parse.py election-day-tweets/election-day-tweets.csv
	python3 parse.py


election-day-tweets/election-day-tweets.csv:
