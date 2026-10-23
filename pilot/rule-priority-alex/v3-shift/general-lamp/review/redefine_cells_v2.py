"""v2: P(template cell has >=10 errors and >=10 non-errors) at 90 test items per template, at the old-protocol rates
(redefine_template_auc_out.txt) and before the option-mass filter, which can only shrink cells."""
import math
q=lambda n,p,k=10: sum(math.comb(n,x)*p**x*(1-p)**(n-x) for x in range(k,n-k+1))
R={"e+":.61,"pi:+":.89,"pi:mod":.50,"pi:num":.30,"view+:digit":.09,"view+:random":.11,"view-num":0.0,"view÷:instead":.21,"view÷:short":.04}
for keep in (1.0,.7,.5):
    n=round(90*keep); tot=0
    print(f"filter keeps {keep:.0%} -> {n} per template")
    for t,p in R.items():
        pr=q(n,p); tot+=pr; print(f"   {t:15s} rate {p:.2f}  P(qualifies) {pr:.2f}")
    print(f"   expected qualifying templates {tot:.1f} of 9")
