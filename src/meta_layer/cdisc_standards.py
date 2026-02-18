"""
CDISC Standards Implementation

Provides CDISC (Clinical Data Interchange Standards Consortium) 
standards compliance for SDTM (Study Data Tabulation Model) and 
ADaM (Analysis Data Model) data structures.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from datetime import datetime
from loguru import logger


class CDISCDomain(Enum):
    """CDISC SDTM domains"""
    DEMOGRAPHICS = "DM"
    SUBJECT_ELEMENTS = "SE"
    TRIAL_ELEMENTS = "TE"
    INCLUSION_EXCLUSION = "IE"
    MEDICATIONS = "CM"
    VITAL_SIGNS = "VS"
    LABORATORY = "LB"
    ADVERSE_EVENTS = "AE"
    DISPOSITION = "DS"
    ECG = "EG"
    PHYSICAL_EXAM = "PE"
    QUESTIONNAIRE = "QS"


@dataclass
class CDISCVariable:
    """CDISC variable definition"""
    name: str
    label: str
    data_type: str
    length: int
    required: bool = True
    controlled_terms: Optional[List[str]] = None
    description: str = ""
    core: str = "REQ"  # REQ, EXP, PERM, SUPP


@dataclass
class CDISCDomainDefinition:
    """CDISC domain definition"""
    domain_code: str
    domain_name: str
    description: str
    variables: Dict[str, CDISCVariable]
    key_variables: List[str] = field(default_factory=list)


class CDISCStandards:
    """CDISC standards implementation"""
    
    def __init__(self, version: str = "3.3"):
        self.version = version
        self.domain_definitions = self._initialize_domain_definitions()
        self.controlled_terminology = self._initialize_controlled_terminology()
    
    def _initialize_domain_definitions(self) -> Dict[str, CDISCDomainDefinition]:
        """Initialize CDISC domain definitions"""
        definitions = {}
        
        # Demographics (DM) domain
        dm_variables = {
            "STUDYID": CDISCVariable("STUDYID", "Study Identifier", "char", 40, True, description="Study identifier"),
            "DOMAIN": CDISCVariable("DOMAIN", "Domain Abbreviation", "char", 2, True, description="Domain abbreviation"),
            "USUBJID": CDISCVariable("USUBJID", "Unique Subject Identifier", "char", 40, True, description="Unique subject identifier"),
            "SUBJID": CDISCVariable("SUBJID", "Subject Identifier for the Study", "char", 20, True, description="Subject identifier"),
            "SITEID": CDISCVariable("SITEID", "Study Site Identifier", "char", 40, True, description="Site identifier"),
            "BRTHDTC": CDISCVariable("BRTHDTC", "Date/Time of Birth", "char", 20, True, description="Date/time of birth"),
            "AGE": CDISCVariable("AGE", "Age", "float", 8, True, description="Age"),
            "AGEU": CDISCVariable("AGEU", "Age Units", "char", 7, True, ["YEARS"], description="Age units"),
            "SEX": CDISCVariable("SEX", "Sex", "char", 1, True, ["M", "F", "U"], description="Sex"),
            "RACE": CDISCVariable("RACE", "Race", "char", 50, True, description="Race"),
            "ETHNIC": CDISCVariable("ETHNIC", "Ethnicity", "char", 50, False, description="Ethnicity"),
            "COUNTRY": CDISCVariable("COUNTRY", "Country", "char", 3, True, description="Country"),
            "ARMCD": CDISCVariable("ARMCD", "Arm Code", "char", 20, True, description="Arm code"),
            "ARM": CDISCVariable("ARM", "Description of Arm", "char", 200, True, description="Arm description"),
            "ACTARMCD": CDISCVariable("ACTARMCD", "Actual Arm Code", "char", 20, False, description="Actual arm code"),
            "ACTARM": CDISCVariable("ACTARM", "Description of Actual Arm", "char", 200, False, description="Actual arm description"),
            "RFICDTC": CDISCVariable("RFICDTC", "Date/Time of Informed Consent", "char", 20, True, description="Date/time of informed consent"),
            "RFENDTC": CDISCVariable("RFENDTC", "Date/Time of End of Participation", "char", 20, False, description="Date/time of end of participation"),
            "RFXSTDTC": CDISCVariable("RFXSTDTC", "Date/Time of First Study Drug Administration", "char", 20, False, description="Date/time of first study drug"),
            "RFXENDTC": CDISCVariable("RFXENDTC", "Date/Time of Last Study Drug Administration", "char", 20, False, description="Date/time of last study drug"),
            "RFSTDTC": CDISCVariable("RFSTDTC", "Subject Start Date/Time", "char", 20, True, description="Subject start date/time"),
            "RFENDTC": CDISCVariable("RFENDTC", "Subject End Date/Time", "char", 20, False, description="Subject end date/time"),
            "RFPENDTC": CDISCVariable("RFPENDTC", "Date/Time of End of Participation", "char", 20, False, description="Date/time of end of participation"),
            "DTHDTC": CDISCVariable("DTHDTC", "Date/Time of Death", "char", 20, False, description="Date/time of death"),
            "DTHFL": CDISCVariable("DTHFL", "Subject Death Flag", "char", 1, True, ["Y", ""], description="Subject death flag"),
            "SITEGR1": CDISCVariable("SITEGR1", "Group 1", "char", 40, False, description="Group 1"),
            "AGEGR1": CDISCVariable("AGEGR1", "Age Group 1", "char", 20, False, description="Age group 1"),
            "RACGR1": CDISCVariable("RACGR1", "Race Group 1", "char", 20, False, description="Race group 1"),
        }
        
        definitions["DM"] = CDISCDomainDefinition(
            domain_code="DM",
            domain_name="Demographics",
            description="Subject demographics and baseline characteristics",
            variables=dm_variables,
            key_variables=["STUDYID", "USUBJID", "DOMAIN"]
        )
        
        # Vital Signs (VS) domain
        vs_variables = {
            "STUDYID": CDISCVariable("STUDYID", "Study Identifier", "char", 40, True, description="Study identifier"),
            "DOMAIN": CDISCVariable("DOMAIN", "Domain Abbreviation", "char", 2, True, description="Domain abbreviation"),
            "USUBJID": CDISCVariable("USUBJID", "Unique Subject Identifier", "char", 40, True, description="Unique subject identifier"),
            "VSSEQ": CDISCVariable("VSSEQ", "Sequence Number", "integer", 8, True, description="Sequence number"),
            "VSTESTCD": CDISCVariable("VSTESTCD", "Vital Signs Test Code", "char", 8, True, description="Vital signs test code"),
            "VSTEST": CDISCVariable("VSTEST", "Vital Signs Test Name", "char", 40, True, description="Vital signs test name"),
            "VSCAT": CDISCVariable("VSCAT", "Category for Vital Signs", "char", 40, False, description="Category for vital signs"),
            "VSORRES": CDISCVariable("VSORRES", "Result or Finding in Original Units", "char", 20, True, description="Result in original units"),
            "VSORRESU": CDISCVariable("VSORRESU", "Original Units", "char", 20, True, description="Original units"),
            "VSSTRESC": CDISCVariable("VSSTRESC", "Character Result/Finding in Standard Format", "char", 200, True, description="Character result in standard format"),
            "VSSTAT": CDISCVariable("VSSTAT", "Completion Status", "char", 1, True, ["DONE", "NOT DONE", "PENDING"], description="Completion status"),
            "VSRESCD": CDISCVariable("VSRESCD", "CDISC Code for Result/Finding", "char", 8, False, description="CDISC code for result"),
            "VSRESU": CDISCVariable("VSRESU", "Standardized Units", "char", 20, True, description="Standardized units"),
            "VISIT": CDISCVariable("VISIT", "Visit Name", "char", 200, True, description="Visit name"),
            "VISITNUM": CDISCVariable("VISITNUM", "Visit Number", "integer", 8, False, description="Visit number"),
            "VISITDY": CDISCVariable("VISITDY", "Study Day of Visit", "integer", 8, True, description="Study day of visit"),
            "VSDTC": CDISCVariable("VSDTC", "Date/Time of Measurements", "char", 20, True, description="Date/time of measurements"),
            "VSDY": CDISCVariable("VSDY", "Study Day of Measurement", "integer", 8, True, description="Study day of measurement"),
            "VSTPT": CDISCVariable("VSTPT", "Planned Time Point Name", "char", 50, False, description="Planned time point name"),
            "VSTPTNUM": CDISCVariable("VSTPTNUM", "Planned Time Point Number", "float", 8, False, description="Planned time point number"),
            "VSELTM": CDISCVariable("VSELTM", "Planned Elapsed Time from Time Point Ref", "char", 6, False, description="Planned elapsed time"),
            "VSBLFL": CDISCVariable("VSBLFL", "Baseline Flag", "char", 1, True, ["Y", ""], description="Baseline flag"),
        }
        
        definitions["VS"] = CDISCDomainDefinition(
            domain_code="VS",
            domain_name="Vital Signs",
            description="Vital signs measurements",
            variables=vs_variables,
            key_variables=["STUDYID", "USUBJID", "DOMAIN", "VSSEQ", "VSTESTCD"]
        )
        
        # Adverse Events (AE) domain
        ae_variables = {
            "STUDYID": CDISCVariable("STUDYID", "Study Identifier", "char", 40, True, description="Study identifier"),
            "DOMAIN": CDISCVariable("DOMAIN", "Domain Abbreviation", "char", 2, True, description="Domain abbreviation"),
            "USUBJID": CDISCVariable("USUBJID", "Unique Subject Identifier", "char", 40, True, description="Unique subject identifier"),
            "AESEQ": CDISCVariable("AESEQ", "Sequence Number", "integer", 8, True, description="Sequence number"),
            "AETERM": CDISCVariable("AETERM", "Reported Term for the Adverse Event", "char", 200, True, description="Reported term for the adverse event"),
            "AELLT": CDISCVariable("AELLT", "LLT Code for Adverse Event", "char", 20, False, description="LLT code for adverse event"),
            "AELLTCD": CDISCVariable("AELLTCD", "LLT Code", "char", 8, False, description="LLT code"),
            "AEDECOD": CDISCVariable("AEDECOD", "Dictionary-Derived Term", "char", 200, True, description="Dictionary-derived term"),
            "AEBODSYS": CDISCVariable("AEBODSYS", "Body System or Organ Class", "char", 200, True, description="Body system or organ class"),
            "AESEV": CDISCVariable("AESEV", "Severity/Intensity", "char", 50, True, ["MILD", "MODERATE", "SEVERE"], description="Severity/intensity"),
            "AESER": CDISCVariable("AESER", "Serious Event Flag", "char", 1, True, ["Y", ""], description="Serious event flag"),
            "AEACN": CDISCVariable("AEACN", "Action Taken with Study Treatment", "char", 20, True, ["NOT CHANGED", "DOSE REDUCED", "DRUG INTERRUPTED", "DRUG WITHDRAWN", "DRUG NOT TAKEN"], description="Action taken with study treatment"),
            "AEREL": CDISCVariable("AEREL", "Relationship of AE to Study Treatment", "char", 20, True, ["RELATED", "POSSIBLY RELATED", "UNLIKELY", "UNRELATED"], description="Relationship of AE to study treatment"),
            "AESTDTC": CDISCVariable("AESTDTC", "Start Date/Time of Adverse Event", "char", 20, True, description="Start date/time of adverse event"),
            "AESTDY": CDISCVariable("AESTDY", "Study Day of Start of Adverse Event", "integer", 8, True, description="Study day of start of adverse event"),
            "AEENDTC": CDISCVariable("AEENDTC", "End Date/Time of Adverse Event", "char", 20, False, description="End date/time of adverse event"),
            "AEENDY": CDISCVariable("AEENDY", "Study Day of End of Adverse Event", "integer", 8, False, description="Study day of end of adverse event"),
            "AEDUR": CDISCVariable("AEDUR", "Duration of Adverse Event", "float", 8, False, description="Duration of adverse event"),
            "AEOUT": CDISCVariable("AEOUT", "Outcome of Adverse Event", "char", 50, True, ["RECOVERED/RESOLVED", "RECOVERING/RESOLVING", "NOT RECOVERED/NOT RESOLVED", "RECOVERED/RESOLVED WITH SEQUELAE", "FATAL", "UNKNOWN"], description="Outcome of adverse event"),
            "AEACNO": CDISCVariable("AEACNO", "Action Taken with Study Treatment (Other)", "char", 255, False, description="Action taken with study treatment (other)"),
            "AETOXGR": CDISCVariable("AETOXGR", "Toxicity Grade", "char", 10, False, description="Toxicity grade"),
        }
        
        definitions["AE"] = CDISCDomainDefinition(
            domain_code="AE",
            domain_name="Adverse Events",
            description="Adverse events data",
            variables=ae_variables,
            key_variables=["STUDYID", "USUBJID", "DOMAIN", "AESEQ"]
        )
        
        return definitions
    
    def _initialize_controlled_terminology(self) -> Dict[str, List[str]]:
        """Initialize controlled terminology"""
        return {
            "SEX": ["M", "F", "U", "UNDIFFERENTIATED"],
            "AGEU": ["YEARS", "MONTHS", "WEEKS", "DAYS"],
            "VSSTAT": ["DONE", "NOT DONE", "PENDING"],
            "AESEV": ["MILD", "MODERATE", "SEVERE"],
            "AESER": ["Y", ""],
            "AEREL": ["RELATED", "POSSIBLY RELATED", "UNLIKELY", "UNRELATED", "NOT ASSESSED"],
            "AEOUT": ["RECOVERED/RESOLVED", "RECOVERING/RESOLVING", "NOT RECOVERED/NOT RESOLVED", 
                     "RECOVERED/RESOLVED WITH SEQUELAE", "FATAL", "UNKNOWN"],
            "AEACN": ["NOT CHANGED", "DOSE REDUCED", "DRUG INTERRUPTED", "DRUG WITHDRAWN", "DRUG NOT TAKEN"],
        }
    
    def get_domain_definition(self, domain_code: str) -> Optional[CDISCDomainDefinition]:
        """Get domain definition by code"""
        return self.domain_definitions.get(domain_code)
    
    def validate_dataset(self, data: pd.DataFrame, domain_code: str) -> Dict[str, Any]:
        """Validate dataset against CDISC standards"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_variables": [],
            "invalid_values": [],
            "data_type_issues": []
        }
        
        domain_def = self.get_domain_definition(domain_code)
        if not domain_def:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Unknown domain: {domain_code}")
            return validation_result
        
        # Check required variables
        for var_name, var_def in domain_def.variables.items():
            if var_def.required and var_def.core == "REQ":
                if var_name not in data.columns:
                    validation_result["missing_variables"].append(var_name)
                    validation_result["valid"] = False
        
        # Check data types and values
        for column in data.columns:
            if column in domain_def.variables:
                var_def = domain_def.variables[column]
                
                # Check controlled terminology
                if var_def.controlled_terms:
                    invalid_values = data[~data[column].isin(var_def.controlled_terms + ["", None])][column].unique()
                    if len(invalid_values) > 0:
                        validation_result["invalid_values"].append({
                            "variable": column,
                            "invalid_values": list(invalid_values),
                            "valid_values": var_def.controlled_terms
                        })
                        validation_result["valid"] = False
        
        return validation_result
    
    def transform_to_cdisc(self, data: pd.DataFrame, domain_code: str, mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Transform data to CDISC format"""
        domain_def = self.get_domain_definition(domain_code)
        if not domain_def:
            logger.error(f"Unknown domain: {domain_code}")
            return data
        
        # Create a copy for transformation
        cdisc_data = data.copy()
        
        # Apply column mapping if provided
        if mapping:
            cdisc_data = cdisc_data.rename(columns=mapping)
        
        # Add required CDISC variables if missing
        for var_name, var_def in domain_def.variables.items():
            if var_def.required and var_name not in cdisc_data.columns:
                if var_name == "DOMAIN":
                    cdisc_data[var_name] = domain_code
                elif var_name == "STUDYID":
                    cdisc_data[var_name] = "STUDY001"  # Default study ID
                elif var_name == "USUBJID" and "patient_id" in cdisc_data.columns:
                    cdisc_data[var_name] = cdisc_data["patient_id"]
                else:
                    cdisc_data[var_name] = ""  # Empty placeholder
        
        # Sort by key variables
        if domain_def.key_variables:
            available_keys = [key for key in domain_def.key_variables if key in cdisc_data.columns]
            if available_keys:
                cdisc_data = cdisc_data.sort_values(available_keys)
        
        logger.info(f"Transformed data to CDISC {domain_code} format")
        return cdisc_data
    
    def get_variable_metadata(self, domain_code: str, variable_name: str) -> Optional[CDISCVariable]:
        """Get variable metadata"""
        domain_def = self.get_domain_definition(domain_code)
        if domain_def and variable_name in domain_def.variables:
            return domain_def.variables[variable_name]
        return None
    
    def list_domains(self) -> List[str]:
        """List all available domains"""
        return list(self.domain_definitions.keys())
    
    def export_domain_definition(self, domain_code: str, output_path: str):
        """Export domain definition to JSON"""
        domain_def = self.get_domain_definition(domain_code)
        if not domain_def:
            logger.error(f"Unknown domain: {domain_code}")
            return
        
        definition_data = {
            "domain_code": domain_def.domain_code,
            "domain_name": domain_def.domain_name,
            "description": domain_def.description,
            "key_variables": domain_def.key_variables,
            "variables": {
                var_name: {
                    "name": var_def.name,
                    "label": var_def.label,
                    "data_type": var_def.data_type,
                    "length": var_def.length,
                    "required": var_def.required,
                    "controlled_terms": var_def.controlled_terms,
                    "description": var_def.description,
                    "core": var_def.core
                }
                for var_name, var_def in domain_def.variables.items()
            }
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(definition_data, f, indent=2)
        
        logger.info(f"Exported {domain_code} domain definition to {output_path}")
