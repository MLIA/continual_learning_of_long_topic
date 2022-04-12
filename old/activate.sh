WHO=$(whoami)
FOLDER=/local/$WHO/libraries
CONDA_BIN=$FOLDER/anaconda3/bin
source $CONDA_BIN/activate
conda activate lire
export PYTHONPATH=$PYTHONPATH:$(pwd)