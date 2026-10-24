set -e
cd /mnt/project-files/pilot/rule-priority-alex/v3-shift/forces-map2/
export MODAL_TOKEN_ID=$modal_token_id MODAL_TOKEN_SECRET=$modal_token_secret SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
T=$1; N=${2:-0}
modal run modal_fm2.py --tag $T --n $N 2>&1 | grep -E "saved|FAILED|Error| / " | awk 'NR%5==1 || /saved|FAILED|Error/' || true
sleep 5; modal run modal_fm2.py --tag $T --get 2>&1 | grep -E " ok|not"
sleep 10; python3 score_fm2.py $T > report_$T.txt; echo scored
