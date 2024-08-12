# Notes about using Donut

## Versions
Specific versions are required for Donut to work
* timm 0.6.13
* pytorch-lightning 1.8.
* transformers 4.25.1
* Numpy pre-2.0.0
* Synthdog requires Pillow 9.5.0, newer versions break handling of fonts
* There are lots of hard to understand error messages and warnings that will appear when using the wrong versions.

## Formatting
* Make sure the dataset and metadata.jsonl files are in the right place
* Image input size must be a multiple of 320
