cd /mnt/project-files/pilot/rule-priority-alex/v3-shift/cicm-goal/
export MODAL_TOKEN_ID=$modal_token_id MODAL_TOKEN_SECRET=$modal_token_secret SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
(modal run modal_cg.py --tag main --step g1 2>&1 | grep -E "saved|FAILED|Error") & (sleep 40; modal run modal_cg.py --tag main --step th 2>&1 | grep -E "saved|FAILED|Error"); wait
modal run modal_cg.py --tag main --step g1 --get 2>&1 | grep -E " ok|not"; modal run modal_cg.py --tag main --step th --get 2>&1 | grep -E " ok|not"
sleep 15; python3 make_cg.py answer main
modal run modal_cg.py --tag main --step g2 2>&1 | grep -E "saved|FAILED|Error"
modal run modal_cg.py --tag main --step g2 --get 2>&1 | grep -E " ok|not"
sleep 15; python3 score_cg.py main > report_main.txt; echo scored
