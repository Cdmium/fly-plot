# A few tools for drosophila image processing

## Installation
1. Install [uv](https://github.com/astral-sh/uv). (slint ui won't work with conda)
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
 - input: a folder of 3 channel larvae image
 - output: a montage png image, a folder of rotated and cropped tiff image