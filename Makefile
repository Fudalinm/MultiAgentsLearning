
LMK=latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error

report:
	cd report && $(LMK) -output-directory=../report-out index.tex

report_watch:
	cd report && $(LMK) -output-directory=../report-out index.tex -pvc

clean:
	cd report && $(LMK) -output-directory=../report-out index.tex -C

.PHONY: report report_watch clean
