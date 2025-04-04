installation

pip install --upgrade pip

pip install -r requirements.txt


to run tests: python -m unittest discover (from top-level directory)


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