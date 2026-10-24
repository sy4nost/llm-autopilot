set -e
cd /mnt/project-files/pilot/rule-priority-alex/v3-shift/goal/
export MODAL_TOKEN_ID=$modal_token_id MODAL_TOKEN_SECRET=$modal_token_secret SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
modal run modal_g.py --tag main --step g1 2>&1 | grep -E "saved|FAILED|Error" || true
modal run modal_g.py --tag main --step g1 --get 2>&1 | grep -E " ok|not"
sleep 10; python3 make_g.py answer main
modal run modal_g.py --tag main --step g2 2>&1 | grep -E "saved|FAILED|Error" || true
modal run modal_g.py --tag main --step g2 --get 2>&1 | grep -E " ok|not"
sleep 10; python3 score_g.py main > report_main.txt; echo scored
