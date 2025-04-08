installation

pip install --upgrade pip

pip install -r requirements.txt


to run tests: python -m unittest discover (from top-level directory)


to run the program (always run the program from the top level directory):
- locally: `python app.py` or `python app.py --mode local`
- remotely: `python app.py --mode remote`


NOTE:
when exporting multiobject segmentation masks, the order of "overlaying" masks is as follows:
1. red
2. blue
3. green
4. orange
5. purple
6. cyan
7. magenta
8. teal
9. black
10. grey
this order is also the order of colours in the pointer colour listbox.




if you come across any problems that are undocumented here, then please visit https://github.com/Chuyun-Shen/SAM_2_Medical_3D/blob/main/INSTALL.md


Documentation:
1. navigate to top-level directory
2. generate documentation via `pydoctor`
windows cmd: `pydoctor --make-html --html-output docs --project-name "Interactive 3D Medical Imaging Segmentation" --docformat=google *.py`

git bash: `pydoctor --make-html --html-output docs --project-name "Interactive 3D Medical Imaging Segmentation" --docformat=google ./*.py`
3. view documentation: a) `cd docs` b) `start index.html`

