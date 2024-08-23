# Variables
PYTHON = python3

# Targets and rules
.PHONY: segmentation measurement clean test

# Target to run the Python script

segmentation:
	$(PYTHON) Locate.py 'Videos/MVI_6495.MP4' 'mask/wormz1.npy' 'mask_figure/worm1.png' 50
	$(PYTHON) Locate.py 'Videos/MVI_6498.MP4' 'mask/wormz2.npy' 'mask_figure/worm2.png' 50
	$(PYTHON) Locate.py 'Videos/MVI_6499.MP4' 'mask/wormz3.npy' 'mask_figure/worm3.png' 50

measurement:

	$(PYTHON) Measure.py 'mask/worm2.npy' 'measurement/wormz2.txt' 13.0643 5


# Target to clean up compiled Python files and other artifacts
clean:
	rm -f *.pyc
	rm -rf __pycache__

# Target to run tests (assuming you have tests)
test:
	pytest tests/
