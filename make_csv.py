"""Convert UCI Statlog german.data (coded A11, A12 ...) into a readable german_credit.csv
(same column names / labels as the OpenML 'credit-g' version)."""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]

cols = ["checking_status", "duration", "credit_history", "purpose", "credit_amount",
        "savings_status", "employment", "installment_commitment", "personal_status",
        "other_parties", "residence_since", "property_magnitude", "age",
        "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
        "own_telephone", "foreign_worker", "class"]

maps = {
    "checking_status": {"A11": "<0", "A12": "0<=X<200", "A13": ">=200", "A14": "no checking"},
    "credit_history": {"A30": "no credits/all paid", "A31": "all paid", "A32": "existing paid",
                       "A33": "delayed previously", "A34": "critical/other existing credit"},
    "purpose": {"A40": "new car", "A41": "used car", "A42": "furniture/equipment",
                "A43": "radio/tv", "A44": "domestic appliance", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining",
                "A49": "business", "A410": "other"},
    "savings_status": {"A61": "<100", "A62": "100<=X<500", "A63": "500<=X<1000",
                       "A64": ">=1000", "A65": "no known savings"},
    "employment": {"A71": "unemployed", "A72": "<1", "A73": "1<=X<4", "A74": "4<=X<7", "A75": ">=7"},
    "personal_status": {"A91": "male div/sep", "A92": "female div/dep/mar", "A93": "male single",
                        "A94": "male mar/wid", "A95": "female single"},
    "other_parties": {"A101": "none", "A102": "co applicant", "A103": "guarantor"},
    "property_magnitude": {"A121": "real estate", "A122": "life insurance", "A123": "car",
                           "A124": "no known property"},
    "other_payment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemp/unskilled non res", "A172": "unskilled resident", "A173": "skilled",
            "A174": "high qualif/self emp/mgmt"},
    "own_telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
    "class": {1: "good", 2: "bad"},
}

df = pd.read_csv(src, header=None, names=cols)
for c, m in maps.items():
    df[c] = df[c].map(m)
assert df.isna().sum().sum() == 0, "unmapped code found"
df.to_csv(dst, index=False)
print(df.shape, df["class"].value_counts().to_dict())
