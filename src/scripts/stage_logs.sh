# list directories in the form "/snapshots/dir/"
for dir in snapshots     
do
    # remove the trailing "/"
    dir=${dir%*/}      
    echo $dir

    git add ${dir}/training_log.csv
    git add ${dir}/corruptions_log.csv
done