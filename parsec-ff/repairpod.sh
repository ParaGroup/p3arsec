#!/bin/bash
#This script fixes errors of the type "*.pod: around line 272: Expected text after =item, not a number
for i in 0 1 2 3 4 5 6 7 8 9 
do
    echo "Replacing '=item $i' to '=item C<$i>'"
    grep -rl "=item $i" * | grep ".*\.pod" | xargs sed -i "s/=item $i/=item C<$i>/g"
done
exit 0