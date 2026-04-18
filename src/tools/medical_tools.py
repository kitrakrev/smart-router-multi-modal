"""Simulated medical tools for demo."""

from __future__ import annotations

from typing import Any


def drug_interaction_check(drug_a: str, drug_b: str) -> dict[str, Any]:
    """Check drug-drug interactions and contraindications."""
    # Simulated interaction database
    interactions = {
        ("warfarin", "aspirin"): {
            "severity": "major",
            "effect": "Increased risk of bleeding",
            "mechanism": "Additive anticoagulant/antiplatelet effects",
            "recommendation": "Avoid combination unless benefit outweighs risk. Monitor INR closely if used together.",
        },
        ("metformin", "lisinopril"): {
            "severity": "minor",
            "effect": "Lisinopril may enhance hypoglycemic effect of metformin",
            "mechanism": "ACE inhibitors may improve insulin sensitivity",
            "recommendation": "Generally safe. Monitor blood glucose during initiation.",
        },
        ("ssri", "maoi"): {
            "severity": "contraindicated",
            "effect": "Risk of serotonin syndrome - potentially fatal",
            "mechanism": "Excessive serotonergic activity",
            "recommendation": "CONTRAINDICATED. Allow 14-day washout between agents.",
        },
        ("simvastatin", "amlodipine"): {
            "severity": "moderate",
            "effect": "Increased risk of myopathy/rhabdomyolysis",
            "mechanism": "CYP3A4 inhibition increases statin levels",
            "recommendation": "Limit simvastatin to 20mg/day when combined with amlodipine.",
        },
    }

    key = (drug_a.lower().strip(), drug_b.lower().strip())
    key_rev = (key[1], key[0])

    result = interactions.get(key) or interactions.get(key_rev)
    if result:
        return {
            "tool": "drug_interaction_check",
            "drug_a": drug_a,
            "drug_b": drug_b,
            **result,
        }
    return {
        "tool": "drug_interaction_check",
        "drug_a": drug_a,
        "drug_b": drug_b,
        "severity": "none_found",
        "effect": "No known interaction found in database",
        "recommendation": "Always verify with a current drug interaction database.",
    }


def clinical_guideline(condition: str) -> dict[str, Any]:
    """Look up clinical practice guidelines."""
    guidelines = {
        "hypertension": {
            "source": "ACC/AHA 2017",
            "classification": "Normal <120/80, Elevated 120-129/<80, Stage 1 130-139/80-89, Stage 2 >=140/>=90",
            "first_line": "ACE inhibitor, ARB, CCB, or thiazide diuretic",
            "lifestyle": "DASH diet, sodium restriction <1500mg/day, exercise 150min/week, weight loss if BMI>25",
            "target": "<130/80 mmHg for most adults",
        },
        "heart failure": {
            "source": "ACC/AHA/HFSA 2022",
            "classification": "HFrEF (EF<=40%), HFmrEF (41-49%), HFpEF (>=50%)",
            "gdmt_hfref": "ACEi/ARB/ARNI + beta-blocker + MRA + SGLT2i (quadruple therapy)",
            "monitoring": "BNP/NT-proBNP, echo q6-12 months, renal function",
            "target": "Symptom improvement, reduce hospitalization, improve survival",
        },
        "diabetes": {
            "source": "ADA 2024 Standards of Care",
            "target_a1c": "<7% for most adults, <8% for elderly/comorbid",
            "first_line": "Metformin + lifestyle modification",
            "second_line": "SGLT2i (if CKD/HF) or GLP-1 RA (if ASCVD/obesity)",
            "monitoring": "A1C every 3-6 months, annual eye/foot exam, eGFR, UACR",
        },
        "colorectal cancer screening": {
            "source": "USPSTF 2021",
            "age_start": "45 years (average risk)",
            "options": "Colonoscopy q10yr, FIT annually, Cologuard q1-3yr, CT colonography q5yr",
            "high_risk": "Earlier screening if FHx, IBD, Lynch syndrome, FAP",
        },
    }

    cond_lower = condition.lower().strip()
    for key, guideline in guidelines.items():
        if key in cond_lower or cond_lower in key:
            return {"tool": "clinical_guideline", "condition": condition, **guideline}

    return {
        "tool": "clinical_guideline",
        "condition": condition,
        "message": "Guideline not found in demo database. Consult UpToDate or relevant society guidelines.",
    }


def lab_reference(test_name: str) -> dict[str, Any]:
    """Look up normal lab value ranges."""
    lab_values = {
        "troponin": {
            "test": "Troponin I (high-sensitivity)",
            "normal_range": "<14 ng/L (female), <22 ng/L (male)",
            "units": "ng/L",
            "critical_high": ">100 ng/L suggests acute MI",
            "note": "Serial measurements recommended (0h, 3h) for rule-out",
        },
        "creatinine": {
            "test": "Serum Creatinine",
            "normal_range": "0.7-1.3 mg/dL (male), 0.6-1.1 mg/dL (female)",
            "units": "mg/dL",
            "critical_high": ">4.0 mg/dL",
            "note": "Use CKD-EPI equation for eGFR calculation",
        },
        "tsh": {
            "test": "Thyroid Stimulating Hormone",
            "normal_range": "0.4-4.0 mIU/L",
            "units": "mIU/L",
            "low": "<0.4 suggests hyperthyroidism",
            "high": ">4.0 suggests hypothyroidism",
            "note": "Check free T4 if TSH abnormal",
        },
        "hemoglobin": {
            "test": "Hemoglobin",
            "normal_range": "13.5-17.5 g/dL (male), 12.0-16.0 g/dL (female)",
            "units": "g/dL",
            "critical_low": "<7.0 g/dL (consider transfusion)",
        },
        "potassium": {
            "test": "Serum Potassium",
            "normal_range": "3.5-5.0 mEq/L",
            "units": "mEq/L",
            "critical_low": "<2.5 mEq/L",
            "critical_high": ">6.5 mEq/L",
            "note": "Check ECG if critically abnormal",
        },
        "glucose": {
            "test": "Fasting Blood Glucose",
            "normal_range": "70-99 mg/dL",
            "units": "mg/dL",
            "prediabetes": "100-125 mg/dL",
            "diabetes": ">=126 mg/dL (confirm with repeat)",
            "critical_low": "<54 mg/dL (severe hypoglycemia)",
        },
        "wbc": {
            "test": "White Blood Cell Count",
            "normal_range": "4,500-11,000 cells/mcL",
            "units": "cells/mcL",
            "low": "<4,000 (leukopenia)",
            "high": ">11,000 (leukocytosis)",
        },
        "inr": {
            "test": "International Normalized Ratio",
            "normal_range": "0.8-1.2 (not on anticoagulation)",
            "therapeutic": "2.0-3.0 (standard anticoagulation), 2.5-3.5 (mechanical valve)",
            "critical_high": ">5.0 (bleeding risk)",
        },
    }

    name_lower = test_name.lower().strip()
    for key, value in lab_values.items():
        if key in name_lower or name_lower in key:
            return {"tool": "lab_reference", "query": test_name, **value}

    return {
        "tool": "lab_reference",
        "query": test_name,
        "message": "Lab reference not found in demo database.",
    }


def dosage_calculator(drug: str, weight_kg: float = 70.0, age_years: int = 40) -> dict[str, Any]:
    """Calculate medication dosages based on weight and age."""
    dosing = {
        "amoxicillin": {
            "adult_dose": "500mg PO TID or 875mg PO BID",
            "pediatric_dose_mg_kg": 25.0,
            "frequency": "every 8 hours",
            "max_daily_mg": 3000,
            "route": "oral",
            "duration": "7-10 days (typical)",
        },
        "ibuprofen": {
            "adult_dose": "200-400mg PO every 4-6 hours",
            "pediatric_dose_mg_kg": 10.0,
            "frequency": "every 6-8 hours",
            "max_daily_mg": 3200,
            "route": "oral",
            "duration": "as needed",
        },
        "metformin": {
            "adult_dose": "500mg PO BID, titrate to 1000mg BID",
            "pediatric_dose_mg_kg": None,
            "frequency": "twice daily with meals",
            "max_daily_mg": 2550,
            "route": "oral",
            "note": "Contraindicated if eGFR <30. Hold for contrast procedures.",
        },
        "vancomycin": {
            "adult_dose": "15-20 mg/kg IV every 8-12 hours",
            "pediatric_dose_mg_kg": 15.0,
            "frequency": "every 6 hours (pediatric), every 8-12 hours (adult)",
            "max_daily_mg": 4000,
            "route": "IV",
            "note": "Monitor trough levels (target 15-20 mcg/mL for serious infections). Adjust for renal function.",
        },
    }

    drug_lower = drug.lower().strip()
    for key, info in dosing.items():
        if key in drug_lower or drug_lower in key:
            result = {"tool": "dosage_calculator", "drug": drug, "weight_kg": weight_kg, "age": age_years, **info}
            # Calculate pediatric dose if applicable
            if age_years < 18 and info.get("pediatric_dose_mg_kg"):
                dose_mg = info["pediatric_dose_mg_kg"] * weight_kg
                result["calculated_dose_mg"] = round(dose_mg, 1)
                result["note"] = f"Calculated: {dose_mg:.1f}mg per dose"
            return result

    return {
        "tool": "dosage_calculator",
        "drug": drug,
        "weight_kg": weight_kg,
        "age": age_years,
        "message": "Drug not found in demo formulary.",
    }


def icd_code_lookup(condition: str) -> dict[str, Any]:
    """Look up ICD-10 diagnostic codes."""
    codes = {
        "type 2 diabetes": {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "category": "E11"},
        "type 1 diabetes": {"code": "E10.9", "description": "Type 1 diabetes mellitus without complications", "category": "E10"},
        "hypertension": {"code": "I10", "description": "Essential (primary) hypertension", "category": "I10"},
        "acute appendicitis": {"code": "K35.80", "description": "Unspecified acute appendicitis without abscess", "category": "K35"},
        "pneumonia": {"code": "J18.9", "description": "Pneumonia, unspecified organism", "category": "J18"},
        "acute myocardial infarction": {"code": "I21.9", "description": "Acute myocardial infarction, unspecified", "category": "I21"},
        "heart failure": {"code": "I50.9", "description": "Heart failure, unspecified", "category": "I50"},
        "asthma": {"code": "J45.909", "description": "Unspecified asthma, uncomplicated", "category": "J45"},
        "depression": {"code": "F32.9", "description": "Major depressive disorder, single episode, unspecified", "category": "F32"},
        "copd": {"code": "J44.1", "description": "Chronic obstructive pulmonary disease with acute exacerbation", "category": "J44"},
        "atrial fibrillation": {"code": "I48.91", "description": "Unspecified atrial fibrillation", "category": "I48"},
        "melanoma": {"code": "C43.9", "description": "Malignant melanoma of skin, unspecified", "category": "C43"},
        "breast cancer": {"code": "C50.919", "description": "Malignant neoplasm of unspecified site of unspecified female breast", "category": "C50"},
    }

    cond_lower = condition.lower().strip()
    for key, value in codes.items():
        if key in cond_lower or cond_lower in key:
            return {"tool": "icd_code_lookup", "condition": condition, **value}

    return {
        "tool": "icd_code_lookup",
        "condition": condition,
        "message": "ICD-10 code not found in demo database. Consult full ICD-10-CM reference.",
    }


# Tool registry for programmatic access
MEDICAL_TOOLS = {
    "drug_interaction_check": {
        "function": drug_interaction_check,
        "description": "Check drug-drug interactions and contraindications",
        "parameters": {"drug_a": "str", "drug_b": "str"},
    },
    "clinical_guideline": {
        "function": clinical_guideline,
        "description": "Look up clinical practice guidelines",
        "parameters": {"condition": "str"},
    },
    "lab_reference": {
        "function": lab_reference,
        "description": "Look up normal lab value ranges",
        "parameters": {"test_name": "str"},
    },
    "dosage_calculator": {
        "function": dosage_calculator,
        "description": "Calculate medication dosages",
        "parameters": {"drug": "str", "weight_kg": "float", "age_years": "int"},
    },
    "icd_code_lookup": {
        "function": icd_code_lookup,
        "description": "Look up ICD-10 diagnostic codes",
        "parameters": {"condition": "str"},
    },
}
