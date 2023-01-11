#!/bin/bash

while getopts g:l:L:a: option; do
    case $option in
	g) gnn=$OPTARG;;
	l) lin1=$OPTARG;;
	L) lin2=$OPTARG;;
	a) attention=$OPTARG;;
    esac
done

cat <<EOF

Convolution hidden dimension: $gnn
MLP hidden dimensions: ($lin1, $lin2)
Attention heads: $attention

EOF

set -e
ENVNAME="pyg-gpu"
ENVDIR=$ENVNAME

cp /staging/groups/burton_group/tmpcody/pyg-gpu.tar.gz .
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

python3 main.py -i cherry_heterodataset.pt -e 100 -k 10 -gh $gnn -lh $lin1 $lin2 -d 0.25 -w 0.0 -a $attention

rm -rf $ENVDIR pyg-gpu.tar.gz
