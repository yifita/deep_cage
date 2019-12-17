#!/bin/bash
# usage: ./run_meshlab_filter INPUT_DIR "*.ply" OUTPUT_DIR MESHLAB_SCRIPT
num_procs=1

inputDir=$1
name="$2"
outputDir="$3"
myDir=$(pwd)
scriptFile="$myDir/$4"
echo "input: $inputDir output: $outputDir extension: $name"

cd $inputDir
find . -type d -exec mkdir -p "$outputDir"/{} \;


function call_meshlab_script () {
	iFile="$1"
	iName="$(basename $iFile)"
	# remove last extension
	iName="${iName%.*}"
	iDir="$(dirname $iFile)"
	oFile="$3/$iName".ply
    sFile="$2"
	# meshlab.meshlabserver -i $iFile -o $oFile -m vn -s $sFile
	echo "meshlab.meshlabserver -i $iFile -o $oFile -s $sFile"
	if [ ! -f "$oFile" ]; then
	# 	# meshlab.meshlabserver -i $iFile -o $oFile -m vn -s $sFile
		meshlab.meshlabserver -i $iFile -o $oFile -s $sFile
	fi
}
export -f call_meshlab_script

echo $scriptFile
find . -type f -wholename "$name"
find . -type f -wholename "$name" | xargs -P $num_procs -I % bash -c 'call_meshlab_script "$@"' _ % $scriptFile $outputDir
cd $myDir
