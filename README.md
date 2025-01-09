# Optical Flow - Classical and Deep Learning Methods

## Dependencies

```python
pip install -r requirements.txt
```

## Lukas-Kanade Optical Flow (Farneback)

> https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method

Run demo with the following command:

```python	
python main_lukas_kanade.py --path=demo-frames/Sintel_alley --path_flo=demo-frames/Sintel_Ground_Truth_alley
```

When the visualization is shown, press a key to continue and see the error metrics.

## RAFT Optical Flow

> https://github.com/princeton-vl/RAFT

Run demo with the following command:

```python
python main_raft.py --model=models/raft-sintel.pth --path=demo-frames/Sintel_alley --path_flo=demo-frames/Sintel_Ground_Truth_alley
```

When the visualization is shown, press a key to continue and see the error metrics.

## Written codes

- `main_lukas_kanade.py` - Lukas-Kanade Optical Flow (Farneback)
- `main_raft.py` - RAFT Optical Flow