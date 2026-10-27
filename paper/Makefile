.PHONY: all assets pdf arxiv clean

all: pdf

assets:
	cd .. && python paper/scripts/make_assets.py

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

arxiv: pdf
	rm -f arxiv_upload.zip
	zip -r arxiv_upload.zip main.tex main.bbl references.bib sections tables/*.tex figures/*.pdf

clean:
	latexmk -C
	rm -f arxiv_upload.zip
