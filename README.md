## Installation

1. Install dependencies.

  - **OSX/Windows** You will probably want to install
    [Anaconda](https://www.continuum.io/downloads#_macosx),
    a Python package manager. ***IMPORTANT:* Install the Python 3.X version.**
  
  - **Linux** You will need Python 3, and the python3 packages `numpy`,
    `matplotlib`, and `pillow`, installed through your package manager or `pip`.

2. Clone this repository.
  
  - Open the terminal, and run `git clone git@github.com:wackywendell/diffusiontrap.git`

That's all, there's no setup for a pure-python program like this.

## Running

1. Export your data.

  - First, you will need to export your data to `tif` format, so that it alternates
yellow and cyan frames, with cyan first.

2. Open a terminal in the right directory.

  - Open the Terminal application.
  
  - Type in `cd /Users/your_username/directory_you_want_to_work_in`
  
  - Run `python3 /path/to/diffusiontrap.py path/to/image.tif`

3. Choose your options.

  - Run `python3 /path/to/diffusiontrap.py --help` to see all the options
  
  - For example, run `python3 /path/to/diffusiontrap.py -t 20 -u 150` to use a
    threshold of 20 and an "upper limit" of 150.

4. Files will appear in the same directory as the original image, with roughly the same name as the original file.

  - The one ending in `.csv` is the spreadsheet with all the columns.
