# A few tools for drosophila image processing

## Installation
1. Install [uv](https://github.com/astral-sh/uv).
2. Clone the repository:
    ```zsh
    git clone https://github.com/BioProgramming-Lab/GUV-tracking
    cd GUV-tracking
    ```
3. Install dependencies:
    ```zsh
    uv sync
    ```

## Usage 
1. larvae-plot
 - input: a folder of 3 channel larvae images
 - output: a montage png image, a folder of rotated and cropped tiff image, a parameter.toml

    ```zsh
    cd directory/of/larvae-plot
    uv run larvae-plot.py larvae.toml # use larvae parameters
    uv run larvae-plot.py adult.toml # use adult parameters
    uv run larvae-plot.py path/to/parameter.toml # use customized parameters
    ```