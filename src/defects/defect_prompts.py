# src/defects/defect_prompts.py
DEFECT_PROMPTS = {
    "no_label": [
        "plastic bucket with no label",
        "unlabeled plastic bucket",
        "bucket without sticker",
        "label missing on plastic bucket",
        "bucket where label is absent"
    ],
    "torn_label": [
        "plastic bucket with torn label",
        "bucket with ripped sticker",
        "damaged label on plastic bucket",
        "label is ripped on the bucket"
    ],
    "folded_label": [
        "plastic bucket with folded label",
        "bucket label is creased",
        "label folded on plastic bucket"
    ],
    "dirty": [
        "plastic bucket with dirty stains",
        "stained plastic bucket",
        "dirty bucket surface"
    ],
    "unreadable_bar_code": [
        "plastic bucket with unreadable bar_code",
        "bar code is unreadable on the bucket",
        "bucket bar code not legible"
    ],
    "no_lid": [
        "plastic bucket without lid",
        "bucket missing its lid",
        "bucket with no top lid"
    ],
    "no_bucket": [
        "no bucket in the image",
        "empty background without bucket",
        "there is no bucket present"
    ],
    "incomplete_view": [
        "bucket partially out of frame",
        "incomplete bucket in the image",
        "bucket cropped and partially visible"
    ],
}
