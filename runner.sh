#!/bin/sh

RUNNER_PY="fl.lstm.py"
ARGS1="127.0.0.1"
ARGS2="ds4"
ARGS3="10"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-1"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-2"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-3"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-5"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-6"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-7"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-8"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-9"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"

cd "/Volumes/beast/Personal/GitHub/Barathwaja/federated-learning/client-10"
python "$RUNNER_PY" "--ip=$ARGS1" "--folder=$ARGS2" "--epochs=$ARGS3"