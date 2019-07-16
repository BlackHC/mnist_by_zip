#!/usr/bin/env bash

test_image=$1

echo 'Compressing training data...' >&2

for train_file in data/*.train; do
    if [[ ! -e ${train_file}.gz ]]; then
        # Keep the original files.
        # Use the best possible compression.
        gzip --best -k $train_file
    fi
done

echo 'Compressing training + test data...' >&2

for train_file in data/*.train; do
    test_file=${train_file/.train/.test}
    cat $train_file $test_image > $test_file

    # Overwrite previously created test files.
    # Use the best possible compression.
    gzip -f --best $test_file
done

echo 'Determining classes using argmin of the compressed size' >&2
compressed_size_by_class=$(
for train_file in data/*.train.gz; do
    digit_class=${train_file#data/digit_}
    digit_class=${digit_class%.train.gz}

    test_file=${train_file/.train/.test}

    printf "%4d   %1d\n" $(( $(stat -c %s $test_file) - $(stat -c %s $train_file) )) $digit_class
done
)

echo 'Size Class' >&2
echo "$compressed_size_by_class" | sort -n >&2
echo >&2

echo "Test file:" ${test_image}
echo "Predicted class:" $(echo "$compressed_size_by_class" | sort -n | head -n 1 | awk '{print $2}')
echo
