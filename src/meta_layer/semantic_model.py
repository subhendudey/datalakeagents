"""
Semantic Model and Clinical Ontology

Defines the semantic structure of clinical trial data using RDF/OWL
and integrates with clinical terminology standards like SNOMED CT, LOINC, etc.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import networkx as nx
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
from loguru import logger


class ClinicalDomain(Enum):
    """Clinical domains for categorizing data"""
    DEMOGRAPHICS = "demographics"
    VITAL_SIGNS = "vital_signs"
    LABORATORY = "laboratory"
    ADVERSE_EVENTS = "adverse_events"
    MEDICATIONS = "medications"
    PROCEDURES = "procedures"
    OUTCOMES = "outcomes"


@dataclass
class SemanticConcept:
    """Represents a clinical concept with semantic properties"""
    concept_id: str
    concept_name: str
    definition: str
    domain: ClinicalDomain
    data_type: str
    terminology: Optional[str] = None  # SNOMED CT, LOINC, etc.
    terminology_code: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)


class ClinicalOntology:
    """Clinical ontology management using RDF/OWL"""
    
    def __init__(self, namespace_uri: str = "http://clinicaltrials.org/ontology#"):
        self.namespace = Namespace(namespace_uri)
        self.graph = Graph()
        self.graph.bind("clinical", self.namespace)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        
        # Initialize core clinical concepts
        self._initialize_core_concepts()
    
    def _initialize_core_concepts(self):
        """Initialize core clinical trial concepts"""
        # Create main classes
        clinical_trial = URIRef(self.namespace + "ClinicalTrial")
        patient = URIRef(self.namespace + "Patient")
        adverse_event = URIRef(self.namespace + "AdverseEvent")
        vital_sign = URIRef(self.namespace + "VitalSign")
        laboratory_test = URIRef(self.namespace + "LaboratoryTest")
        
        # Add classes to ontology
        self.graph.add((clinical_trial, RDF.type, OWL.Class))
        self.graph.add((patient, RDF.type, OWL.Class))
        self.graph.add((adverse_event, RDF.type, OWL.Class))
        self.graph.add((vital_sign, RDF.type, OWL.Class))
        self.graph.add((laboratory_test, RDF.type, OWL.Class))
        
        # Add properties
        has_patient = URIRef(self.namespace + "hasPatient")
        has_adverse_event = URIRef(self.namespace + "hasAdverseEvent")
        has_vital_sign = URIRef(self.namespace + "hasVitalSign")
        has_lab_result = URIRef(self.namespace + "hasLaboratoryResult")
        
        self.graph.add((has_patient, RDF.type, OWL.ObjectProperty))
        self.graph.add((has_adverse_event, RDF.type, OWL.ObjectProperty))
        self.graph.add((has_vital_sign, RDF.type, OWL.ObjectProperty))
        self.graph.add((has_lab_result, RDF.type, OWL.ObjectProperty))
    
    def add_concept(self, concept: SemanticConcept):
        """Add a clinical concept to the ontology"""
        concept_uri = URIRef(self.namespace + concept.concept_id)
        
        # Add concept as class or individual
        self.graph.add((concept_uri, RDF.type, OWL.Class))
        self.graph.add((concept_uri, RDFS.label, Literal(concept.concept_name)))
        self.graph.add((concept_uri, RDFS.comment, Literal(concept.definition)))
        
        # Add domain information
        domain_uri = URIRef(self.namespace + concept.domain.value)
        self.graph.add((concept_uri, RDFS.subClassOf, domain_uri))
        
        # Add terminology mapping if available
        if concept.terminology and concept.terminology_code:
            terminology_uri = URIRef(f"http://terminology.org/{concept.terminology.lower()}/{concept.terminology_code}")
            self.graph.add((concept_uri, OWL.sameAs, terminology_uri))
        
        # Add synonyms
        for synonym in concept.synonyms:
            self.graph.add((concept_uri, RDFS.seeAlso, Literal(synonym)))
        
        logger.info(f"Added concept {concept.concept_name} to ontology")
    
    def get_concept_relationships(self, concept_id: str) -> Dict[str, List[str]]:
        """Get all relationships for a concept"""
        concept_uri = URIRef(self.namespace + concept_id)
        relationships = {}
        
        for predicate, obj in self.graph.triples((concept_uri, None, None)):
            pred_name = str(predicate).split("#")[-1]
            if pred_name not in relationships:
                relationships[pred_name] = []
            relationships[pred_name].append(str(obj))
        
        return relationships
    
    def query_concepts_by_domain(self, domain: ClinicalDomain) -> List[SemanticConcept]:
        """Query all concepts in a specific domain"""
        domain_uri = URIRef(self.namespace + domain.value)
        concepts = []
        
        for subject in self.graph.subjects(RDFS.subClassOf, domain_uri):
            concept_id = str(subject).split("#")[-1]
            concept_name = str(self.graph.value(subject, RDFS.label))
            definition = str(self.graph.value(subject, RDFS.comment))
            
            concept = SemanticConcept(
                concept_id=concept_id,
                concept_name=concept_name,
                definition=definition,
                domain=domain,
                data_type="string"  # Default
            )
            concepts.append(concept)
        
        return concepts
    
    def export_ontology(self, file_path: str, format: str = "turtle"):
        """Export ontology to file"""
        self.graph.serialize(destination=file_path, format=format)
        logger.info(f"Ontology exported to {file_path}")
    
    def import_ontology(self, file_path: str, format: str = "turtle"):
        """Import ontology from file"""
        self.graph.parse(file_path, format=format)
        logger.info(f"Ontology imported from {file_path}")


class SemanticModel:
    """High-level semantic model for clinical trial data"""
    
    def __init__(self, ontology_path: Optional[str] = None):
        self.ontology = ClinicalOntology()
        self.concepts: Dict[str, SemanticConcept] = {}
        self.relationship_graph = nx.DiGraph()
        
        if ontology_path:
            self.ontology.import_ontology(ontology_path)
        
        self._initialize_standard_concepts()
    
    def _initialize_standard_concepts(self):
        """Initialize standard clinical trial concepts"""
        standard_concepts = [
            SemanticConcept(
                concept_id="patient_id",
                concept_name="Patient Identifier",
                definition="Unique identifier for a patient in the clinical trial",
                domain=ClinicalDomain.DEMOGRAPHICS,
                data_type="string",
                constraints={"required": True, "unique": True}
            ),
            SemanticConcept(
                concept_id="age",
                concept_name="Age",
                definition="Age of the patient at trial enrollment",
                domain=ClinicalDomain.DEMOGRAPHICS,
                data_type="integer",
                constraints={"min_value": 0, "max_value": 150}
            ),
            SemanticConcept(
                concept_id="gender",
                concept_name="Gender",
                definition="Biological gender of the patient",
                domain=ClinicalDomain.DEMOGRAPHICS,
                data_type="categorical",
                terminology="SNOMED CT",
                terminology_code="263495000",
                synonyms=["sex"]
            ),
            SemanticConcept(
                concept_id="blood_pressure_systolic",
                concept_name="Systolic Blood Pressure",
                definition="Systolic blood pressure measurement in mmHg",
                domain=ClinicalDomain.VITAL_SIGNS,
                data_type="float",
                terminology="LOINC",
                terminology_code="8480-6",
                constraints={"min_value": 0, "max_value": 300}
            ),
            SemanticConcept(
                concept_id="blood_pressure_diastolic",
                concept_name="Diastolic Blood Pressure", 
                definition="Diastolic blood pressure measurement in mmHg",
                domain=ClinicalDomain.VITAL_SIGNS,
                data_type="float",
                terminology="LOINC",
                terminology_code="8462-4",
                constraints={"min_value": 0, "max_value": 200}
            ),
            SemanticConcept(
                concept_id="adverse_event_severity",
                concept_name="Adverse Event Severity",
                definition="Severity grade of an adverse event",
                domain=ClinicalDomain.ADVERSE_EVENTS,
                data_type="categorical",
                terminology="CTCAE",
                terminology_code="Grade",
                constraints={"values": ["Mild", "Moderate", "Severe", "Life-threatening"]}
            )
        ]
        
        for concept in standard_concepts:
            self.add_concept(concept)
    
    def add_concept(self, concept: SemanticConcept):
        """Add a concept to the semantic model"""
        self.concepts[concept.concept_id] = concept
        self.ontology.add_concept(concept)
        
        # Add to relationship graph
        self.relationship_graph.add_node(
            concept.concept_id,
            name=concept.concept_name,
            domain=concept.domain.value,
            data_type=concept.data_type
        )
        
        # Add relationships
        for rel_type, target_concept in concept.relationships.items():
            self.relationship_graph.add_edge(concept.concept_id, target_concept, type=rel_type)
    
    def get_concept(self, concept_id: str) -> Optional[SemanticConcept]:
        """Get a concept by ID"""
        return self.concepts.get(concept_id)
    
    def get_concepts_by_domain(self, domain: ClinicalDomain) -> List[SemanticConcept]:
        """Get all concepts in a specific domain"""
        return [concept for concept in self.concepts.values() if concept.domain == domain]
    
    def validate_data_semantics(self, data: Dict[str, Any], required_concepts: List[str]) -> Dict[str, Any]:
        """Validate data against semantic model"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_concepts": []
        }
        
        # Check required concepts
        for concept_id in required_concepts:
            if concept_id not in self.concepts:
                validation_results["missing_concepts"].append(concept_id)
                validation_results["valid"] = False
                continue
            
            concept = self.concepts[concept_id]
            
            # Check if concept exists in data
            if concept_id not in data:
                validation_results["errors"].append(f"Missing required concept: {concept.concept_name}")
                validation_results["valid"] = False
                continue
            
            # Validate data type
            value = data[concept_id]
            if not self._validate_data_type(value, concept.data_type):
                validation_results["errors"].append(
                    f"Invalid data type for {concept.concept_name}: expected {concept.data_type}"
                )
                validation_results["valid"] = False
            
            # Validate constraints
            constraint_errors = self._validate_constraints(value, concept.constraints)
            validation_results["errors"].extend(constraint_errors)
            if constraint_errors:
                validation_results["valid"] = False
        
        return validation_results
    
    def _validate_data_type(self, value: Any, expected_type: str) -> bool:
        """Validate data type"""
        type_validators = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "boolean": lambda x: isinstance(x, bool),
            "categorical": lambda x: isinstance(x, str),
            "date": lambda x: hasattr(x, 'strftime') or isinstance(x, str)
        }
        
        validator = type_validators.get(expected_type, lambda x: True)
        return validator(value)
    
    def _validate_constraints(self, value: Any, constraints: Dict[str, Any]) -> List[str]:
        """Validate value against constraints"""
        errors = []
        
        if "min_value" in constraints and value < constraints["min_value"]:
            errors.append(f"Value {value} is below minimum {constraints['min_value']}")
        
        if "max_value" in constraints and value > constraints["max_value"]:
            errors.append(f"Value {value} is above maximum {constraints['max_value']}")
        
        if "values" in constraints and value not in constraints["values"]:
            errors.append(f"Value {value} not in allowed values: {constraints['values']}")
        
        return errors
    
    def get_semantic_path(self, source_concept: str, target_concept: str) -> Optional[List[str]]:
        """Find semantic path between two concepts"""
        try:
            return nx.shortest_path(self.relationship_graph, source_concept, target_concept)
        except nx.NetworkXNoPath:
            return None
    
    def export_semantic_model(self, file_path: str):
        """Export semantic model to JSON"""
        model_data = {
            "concepts": {
                concept_id: {
                    "concept_name": concept.concept_name,
                    "definition": concept.definition,
                    "domain": concept.domain.value,
                    "data_type": concept.data_type,
                    "terminology": concept.terminology,
                    "terminology_code": concept.terminology_code,
                    "synonyms": concept.synonyms,
                    "relationships": concept.relationships,
                    "constraints": concept.constraints
                }
                for concept_id, concept in self.concepts.items()
            },
            "relationships": [
                {"source": source, "target": target, "type": data["type"]}
                for source, target, data in self.relationship_graph.edges(data=True)
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Semantic model exported to {file_path}")
