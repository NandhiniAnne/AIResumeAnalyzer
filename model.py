from huggingface_hub import snapshot_download
import spacy

# Download the model from the Hub
model_path = snapshot_download("amjad-awad/skill-extractor", repo_type="model")

# Load the model with spaCy
nlp = spacy.load(model_path)

# Example usage
text = """Technical Skillset
 Blockchain | Digital Assets | Sitecore | Backbase | SAP – FI, CO, SD, MM, PP | S/4 HANA
 Responsive Web | Mobile Apps | Clarity | ServiceNow | EPIC | ETL | Big Data | Hadoop
 AWS | Azure | Azure DevOps (TFS) | Docker | Kubernetes | Mesos | GIT | Subversion
 Oracle Cloud | TIBCO | Salesforce | Microsoft Dynamics | Comburent | Pega DSM
 Office 365 | Microsoft Office Suite | Visio | Adobe Workfront | Hyperion | Jira | Coupa
 IBM Identity Security Access Manager | Shape Security | Cloudflare | Key Cloak | BioCatch
"""
doc = nlp(text)

# Extract skill entities
skills = [ent.text for ent in doc.ents if "SKILLS" in ent.label_]
print(skills)

