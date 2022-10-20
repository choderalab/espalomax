for big_index in {6..25}
do
    export BIG_INDEX=$big_index
    bsub < grab.sh
done
