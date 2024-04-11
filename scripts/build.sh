#!/bin/bash

set -e

#### parse augments
CLEAR_BEFORE_BUILD="off"
BUILD_LOG="on"
RUN_TEST="off"
BUILD_DEBUG="off"
for i in "$@"; do
    case $i in
        --nolog) BUILD_LOG='off' ;;
        --test) RUN_TEST="on" ;;
        --debug) BUILD_DEBUG="on" ;;
        --clear) CLEAR_BEFORE_BUILD="on" ;;
        -h) echo "USAGE: build.sh [-v]"; exit 0 ;;
        *) echo "Unknown option \`$i\`"; exit 1 ;;
    esac
done
DEFINATION="-DBUILD_LOG=$BUILD_LOG -DBUILD_DEBUG=$BUILD_DEBUG"

#### get workspace directory
if [ "$(uname)" == "Darwin" ];
then
    SCRIPT_DIR=$(greadlink -f $(dirname $0)) # brew install coreutils
    WORKSPACE_DIR=$(dirname $SCRIPT_DIR)
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ];
then
    SCRIPT_DIR=$(readlink -f $(dirname $0))
    WORKSPACE_DIR=$(dirname $SCRIPT_DIR)
else
    echo "Unsupported OS"
    exit 1
fi

if [ "$CLEAR_BEFORE_BUILD" == "on" ]; then
    echo -e "\033[1;36m++ clear \033[m"
    rm -rf $WORKSPACE_DIR/.build || true
fi

mkdir -p $WORKSPACE_DIR/.build
mkdir -p $WORKSPACE_DIR/.temp

#### build
echo -e "\033[1;36m++ build \033[m"
sh -c "cd $WORKSPACE_DIR/.build && \
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 $DEFINATION .."
cp $WORKSPACE_DIR/.build/compile_commands.json $WORKSPACE_DIR || true
sh -c "cd $WORKSPACE_DIR/.build && make -j8 || true"

#### test
if [ "$RUN_TEST" == "on" ]; then
    echo -e "\033[1;36m++ test \033[m"
    
    if [ "$(uname)" == "Darwin" ];
    then
        TESTS=$(find $WORKSPACE_DIR/.build/test -type f -perm +111 -name "test_*")
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ];
    then
        TESTS=$(find $WORKSPACE_DIR/.build/test -type f -executable -name "test_*")
    fi
    
    for test in $TESTS; do
        echo -e "\033[0;35m-- run $test \033[m"
        $test
    done
fi