# Data Directory

This directory contains the dataset and related files for the ML Text Sentiment Analysis project.

## Dataset Information

### IMDB Movie Reviews Dataset

The project uses the IMDB Movie Reviews dataset for binary sentiment classification.

- **Task**: Binary sentiment classification (positive/negative)
- **Size**: 50,000 movie reviews
- **Split**: 25,000 training samples, 25,000 test samples
- **Classes**: Balanced (50% positive, 50% negative)

### Dataset Structure

After preprocessing, the data will be split into:
- `train.csv` - Training set (60% of data)
- `val.csv` - Validation set (20% of data) 
- `test.csv` - Test set (20% of data)

Each CSV file contains:
- `review` - The movie review text
- `label` - Binary label (0=negative, 1=positive)

## Download Instructions

### Option 1: Using the existing dataset in the project

If you already have `IMDB Dataset.csv` in the main project directory:

1. The data loading script will automatically use this file
2. Run the data processing script:
   ```bash
   python src/data.py
   ```

### Option 2: Download from Kaggle

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle credentials**:
   - Go to your [Kaggle account settings](https://www.kaggle.com/account)
   - Create a new API token (downloads `kaggle.json`)
   - Place the file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

3. **Download the dataset**:
   ```bash
   kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   unzip imdb-dataset-of-50k-movie-reviews.zip
   ```

### Option 3: Manual Download

1. Go to the [Kaggle dataset page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Click "Download" to get the ZIP file
3. Extract `IMDB Dataset.csv` to the project root directory

### Option 4: Using TensorFlow Datasets (Alternative)

If you prefer to use TensorFlow's built-in dataset:

```python
import tensorflow_datasets as tfds

# Load IMDB reviews dataset
(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)
```

## Data Preparation

After downloading the dataset, run the data preparation script:

```bash
# From the project root directory
python src/data.py
```

This will:
1. Load and clean the raw dataset
2. Remove duplicates and very short reviews
3. Create balanced train/validation/test splits
4. Save the processed data to CSV files in this directory

## File Descriptions

After running the data preparation:

- `train.csv` - Training set for model training
- `val.csv` - Validation set for hyperparameter tuning
- `test.csv` - Test set for final evaluation
- `IMDB Dataset.csv` - Original raw dataset (if downloaded)

## Data Statistics

Expected statistics after preprocessing:

- **Total samples**: ~50,000 (after removing duplicates)
- **Average review length**: ~230 words
- **Vocabulary size**: ~88,000 unique words
- **Class distribution**: Balanced (50%/50%)

### Split Sizes (Approximate)
- Training: ~30,000 samples (60%)
- Validation: ~10,000 samples (20%)
- Test: ~10,000 samples (20%)

## Data Quality

The preprocessing pipeline performs:

- **HTML tag removal**: Strips HTML formatting
- **Duplicate removal**: Removes duplicate reviews
- **Length filtering**: Removes very short reviews (<10 characters)
- **Text normalization**: Basic cleaning and formatting
- **Balanced sampling**: Ensures equal representation of classes

## Usage Notes

1. **Large Files**: The dataset files are not included in version control (see `.gitignore`)
2. **Reproducibility**: Random seed (42) is used for consistent splits
3. **Memory Usage**: Full dataset requires ~200MB RAM for processing
4. **Processing Time**: Data preparation takes 1-2 minutes on modern hardware

## Troubleshooting

### Common Issues

1. **File not found error**:
   - Ensure `IMDB Dataset.csv` is in the project root directory
   - Check file name spelling and capitalization

2. **Memory issues**:
   - Close other applications to free up RAM
   - Use a subset of data for testing: modify `data.py`

3. **Permission errors**:
   - Ensure write permissions for the data directory
   - Check if files are being used by other programs

4. **Download issues**:
   - Verify internet connection
   - Check Kaggle API credentials
   - Try manual download as fallback

### Getting Help

If you encounter issues:
1. Check the error logs in the console
2. Verify all dependencies are installed: `pip install -r requirements.txt`
3. Ensure you have sufficient disk space (~1GB recommended)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```