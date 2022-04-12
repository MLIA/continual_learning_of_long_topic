WHO=$(whoami)
FOLDER=/local/$WHO/libraries
CONDA_BIN=$FOLDER/anaconda3/bin/conda
mkdirs $FOLDER
bash /net/sundays/gerald/download/Anaconda3-2020.11-Linux-x86_64.sh --prefix $FOLDER/anaconda3
$CONDA_BIN init bash
echo "You should restart your terminal and run 'configure.sh'"