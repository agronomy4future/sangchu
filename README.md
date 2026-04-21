# sangchu

Green leaf area measurement for greenhouse pot crops using top-view images.

## Installation

```bash
pip install git+https://github.com/agronomy4future/sangchu
```

## Usage

```python
from sangchu import greenhousepot

df = greenhousepot(
    input_folder  = "./images",
    output_folder = "./results",
    ruler_cm      = 23,
    min_leaf_area = 5.0,
)
print(df)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_folder` | str | required | Path to folder containing `.jpg` images |
| `output_folder` | str | required | Path to folder where results will be saved |
| `ruler_cm` | float | `23` | Actual length of the scale bar in the image (cm) |
| `min_leaf_area` | float | `5.0` | Minimum leaf area (cm²). Smaller clusters are ignored |

## Returns

`pandas.DataFrame` with columns:

| Column | Description |
|---|---|
| File Name | Image filename |
| Leaf Area (cm2) | Detected green leaf area (cm²) |
| Green Ratio (%) | Green area as % of ROI |
| Scale (cm/pixel) | Pixel-to-cm conversion ratio |
| Min Leaf Area (cm2) | Minimum leaf area threshold used |
| Image Path | Full path to input image |

Results are also saved as `analysis_results.csv` in the output folder.

## Output Files

For each image, three files are saved in `output_folder`:

- `*_analyzed.jpg` — original image with green overlay and contour
- `*_mask.jpg` — binary green mask
- `analysis_results.csv` — all results in one CSV

## How It Works

1. **ROI selection** — left-click 4 corner points of the analysis area
2. **Scale selection** — left-click 2 endpoints of the ruler/scale bar
3. **Green detection** — HSV + ExG + LAB a* combination
4. **Background removal** — white paper, metal grid, yellow stickers automatically excluded
5. **Largest cluster** — only the largest green cluster is used (one plant per pot)

## Controls

| Key | Action |
|---|---|
| Left-click | Add point |
| BACKSPACE | Remove last point |
| `+` / `-` | Zoom in / out |
| ENTER | Confirm selection |
| ESC | Skip image |

## Requirements

- Python ≥ 3.8
- opencv-python
- numpy
- pandas

## License

MIT License — © agronomy4future
