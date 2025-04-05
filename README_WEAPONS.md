# Weapons Detection Enhancement

This enhancement adds specialized weapon detection capabilities to the Border Surveillance System, with specific classification for:
- Rifles
- Bazookas/Launchers
- Shotguns
- Handguns
- Knives
- Grenades
- Generic weapons (other types)

## Training the Weapons Detection Model

### Prerequisites
- Python 3.8+
- Kaggle API credentials (for downloading datasets)
- PyTorch and YOLOv8 (included in requirements.txt)

### Setup Kaggle API
1. Create a Kaggle account if you don't have one at [kaggle.com](https://www.kaggle.com/)
2. Generate an API token by going to your account settings and clicking "Create New API Token"
3. This will download a `kaggle.json` file with your credentials
4. Place this file in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
5. Set appropriate permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

### Train the Model
Run the training script:

```bash
python src/train_weapons_model.py
```

Options:
- `--epochs`: Number of training epochs (default: 30)
- `--batch`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--no-kaggle`: Skip Kaggle dataset download (use if you've already downloaded the dataset)

Example:
```bash
python src/train_weapons_model.py --epochs 50 --batch 8
```

The script will:
1. Download a weapons dataset from Kaggle
2. Prepare the dataset in YOLOv8 format
3. Train a YOLOv8 model on the dataset
4. Save the trained model to `models/weapons_detector.pt`

### Training Details
- The model is fine-tuned from a pre-trained YOLOv8 model
- Training uses transfer learning to adapt to weapon detection
- The dataset is split 80/20 for training and validation
- Early stopping is used to prevent overfitting

## Using the Weapons Detection Model

The Border Surveillance System will automatically use the specialized weapons model if it exists. No additional configuration is needed.

The detection system:
1. Uses the trained model to detect and classify weapons
2. Generates alerts with specific weapon types
3. Displays specialized visual indicators for different weapon categories

## Weapon Classes and Alert Thresholds

Each weapon type has a different alert threshold based on its danger level:

| Weapon Type | Alert Threshold | Description |
|-------------|-----------------|-------------|
| Bazooka     | 0.40           | Highest priority - serious threat |
| Grenade     | 0.40           | Explosive device - high priority |
| Rifle       | 0.45           | Long firearm - high priority |
| Shotgun     | 0.45           | High damage potential |
| Handgun     | 0.50           | Concealable firearm |
| Knife       | 0.55           | Bladed weapon |
| Generic     | 0.60           | Other weapons |

## Customizing the Model

To use a different dataset or model architecture:

1. Modify the `dataset_slug` in `src/train_weapons_model.py` to use a different Kaggle dataset
2. Update the `class_mapping` in the script to match your dataset structure
3. Adjust the weapon classes in `WEAPON_CLASSES` to include additional categories

## Testing the Model

To test the model on specific video files:
1. Run the system with a video file input using the UI toggle button
2. Or specify a video file directly: `python src/run.py -s path/to/video.mp4`

## Known Limitations

- Best results are achieved with clear, unobstructed views of weapons
- Detection accuracy may be reduced in low light conditions 
- Some harmless objects may trigger false positives (especially for the generic weapons class)
- The model requires sufficient computational resources for real-time processing 