WHO=$(whoami)
FOLDER="/local/$WHO/libraries"
CONDA_BIN="$FOLDER/anaconda3/bin"
echo "source $CONDA_BIN/activate"
source "$CONDA_BIN/activate"
conda env create -f environment.yml

