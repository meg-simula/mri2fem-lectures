default:
	pdflatex lectures5-6.tex

watch:
	fswatch -o lectures5-6.tex | xargs -n1 -I{} pdflatex lectures5-6.tex
clean:
	rm *.aux *.nav *.log *.out *.snm *.toc *.vrb *~
