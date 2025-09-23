# skill_taxonomy.py
"""
Comprehensive skill taxonomy + categorization helpers.

Usage:
    from skill_taxonomy import categorize_skills_for_resume, SKILL_TAXONOMY

    categorized = categorize_skills_for_resume(extracted_skills, full_text="<full resume text>")
    # categorized is a dict of categories -> [skills], plus "uncategorized"
"""

import re
from collections import defaultdict
from difflib import get_close_matches

# ---------------------------
# Canonical taxonomy (extensive, but not exhaustive)
# Add/remove items to extend.
# All values are lowercase canonical tokens.
# ---------------------------
SKILL_TAXONOMY = {
    "programming_languages": [
        "python","java","javascript","typescript","c#","c++","c","go","rust","ruby","php","swift","kotlin","r","scala","perl","matlab","haskell","lua","objective-c"
    ],
    "frameworks_libraries": [
        "react","angular","vue","ember","svelte","next.js","nuxt","gatsby",
        "spring","spring boot","hibernate","django","flask","express","laravel",
        ".net","asp.net","dotnet core","dotnet","rails","phoenix","play framework",
        "tensorflow","pytorch","scikit-learn","keras","mxnet"
    ],
    "frontend": [
        "html","html5","css","css3","sass","less","bootstrap","tailwind","jquery"
    ],
    "mobile": [
        "react native","flutter","android","ios","swiftui","xamarin","cordova"
    ],
    "databases": [
        "mysql","postgresql","postgres","oracle","sql server","mssql","mongodb","redis","cassandra","dynamodb","couchbase","sqlite","teradata","snowflake","bigquery"
    ],
    "data_engineering": [
        "spark","pyspark","hadoop","hive","pig","airflow","kafka","flink","databricks","etl","etl pipeline","sqoop"
    ],
    "data_science_ml": [
        "pandas","numpy","scipy","scikit-learn","xgboost","lightgbm","catboost","nlp","computer vision","cv","transformers","huggingface"
    ],
    "cloud_platforms": [
        "aws","amazon web services","azure","gcp","google cloud","ibm cloud","oracle cloud","digitalocean"
    ],
    "cloud_services_aws": [
        "ec2","s3","lambda","rds","dynamodb","ecs","eks","cloudformation","iam","cloudwatch","route53"
    ],
    "containers_orchestration": [
        "docker","kubernetes","k8s","mesos","nomad"
    ],
    "ci_cd": [
        "jenkins","gitlab ci","github actions","circleci","bamboo","teamcity","azure devops","travis ci"
    ],
    "infrastructure_as_code": [
        "terraform","ansible","puppet","chef","cloudformation"
    ],
    "monitoring_logging": [
        "prometheus","grafana","elk","elasticsearch","logstash","kibana","splunk","newrelic","datadog"
    ],
    "testing_tools": [
        "jest","mocha","pytest","junit","selenium","cucumber","robot framework","karma","enzyme"
    ],
    "message_brokers_streaming": [
        "rabbitmq","activemq","kafka","aws sns","aws sqs","nats","google pubsub"
    ],
    "big_data_tools": [
        "hadoop","spark","hive","hbase","impala","tez","oozie"
    ],
    "analytics_visualization": [
        "tableau","power bi","looker","qlik","superset","matplotlib","seaborn","d3.js"
    ],
    "devops_tools": [
        "git","github","gitlab","bitbucket","docker","ansible","kubernetes","helm","vault"
    ],
    "security": [
        "cissp","cisa","cism","security","owasp","saml","oauth","jwt","ssl","tls","vulnerability","penetration testing","pentest"
    ],
    "networking": [
        "tcp/ip","dns","http","https","load balancing","vpn","firewall","routing","switching"
    ],
    "ide_tools": [
        "vscode","visual studio","intellij","pycharm","eclipse","netbeans","android studio","xcode"
    ],
    "version_control": [
        "git","svn","mercurial"
    ],
    "project_management": [
        "agile","scrum","kanban","waterfall","jira","confluence","asana","trello","microsoft project"
    ],
    "methodologies": [
        "tdd","bdd","ddd","ci/cd","devops","site reliability engineering","sre","test driven development"
    ],
    "office_productivity": [
        "excel","powerpoint","word","google sheets","google docs","slack","zoom","office 365"
    ],
    "certifications": [
        "aws certified solutions architect","aws certified developer","gcp professional data engineer","pmp","cissp","azure fundamentals"
    ],
    "soft_skills": [
        "communication","leadership","teamwork","management","problem solving","time management","adaptability","critical thinking","presentation","mentoring"
    ],
    "graphics_design": [
        "photoshop","illustrator","figma","sketch","xd"
    ],
    "misc_tools": [
        "unix","linux","bash","powershell","shell scripting","windows","macos"
    ]
}

# Flatten taxonomy lookup table for quick matching (value->category map)
_FLAT_SKILL_TO_CATEGORY = {}
for cat, toks in SKILL_TAXONOMY.items():
    for t in toks:
        _FLAT_SKILL_TO_CATEGORY[t.lower()] = cat

# small synonyms map (common abbreviations -> canonical)
SYNONYM_MAP = {
    "k8s": "kubernetes",
    "js": "javascript",
    "py": "python",
    "tf": "tensorflow",
    "s3": "s3",
    "ec2": "ec2",
    "rds": "rds",
    "db": "database",
    "sql server": "mssql",
    "postgres": "postgresql",
    "node": "node.js",
    "nodejs": "node.js",
    "net core": ".net core",
    ".net core": "dotnet core",
    "vuejs": "vue",
    "reactjs": "react",
    "aws lambda": "lambda",
}

# Useful helper to normalize strings
def normalize_skill_string(s: str) -> str:
    if not s:
        return ""
    s2 = s.lower().strip()
    s2 = re.sub(r'[\u2013\u2014]', '-', s2)  # normalize dashes
    s2 = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', s2)  # strip non-alnum edges
    s2 = re.sub(r'\s+', ' ', s2).strip()
    # map synonyms
    if s2 in SYNONYM_MAP:
        return SYNONYM_MAP[s2]
    return s2

# Main categorization function
def categorize_skills(extracted_skills, full_text=None, fuzzy_cutoff=0.7):
    """
    extracted_skills: list of strings (raw normalized strings recommended)
    full_text: optional full resume text for context-based scanning (not used by default)
    Returns: (categorized_dict, uncategorized_list)
      categorized_dict: category -> [skill strings (canonical)]
    """
    categorized = defaultdict(list)
    uncategorized = []
    # prepare list of known keys
    known_tokens = list(_FLAT_SKILL_TO_CATEGORY.keys())

    for raw in extracted_skills or []:
        s = normalize_skill_string(raw)
        if not s:
            continue
        matched = False
        # exact match
        if s in _FLAT_SKILL_TO_CATEGORY:
            cat = _FLAT_SKILL_TO_CATEGORY[s]
            categorized[cat].append(s)
            matched = True
            continue
        # substring match (e.g., "react.js" contains "react")
        for key in known_tokens:
            if key in s or s in key:
                cat = _FLAT_SKILL_TO_CATEGORY[key]
                categorized[cat].append(s)
                matched = True
                break
        if matched:
            continue
        # fuzzy match (difflib)
        close = get_close_matches(s, known_tokens, n=1, cutoff=fuzzy_cutoff)
        if close:
            cat = _FLAT_SKILL_TO_CATEGORY[close[0]]
            categorized[cat].append(s)
            matched = True
            continue
        # not matched -> uncategorized for manual review
        uncategorized.append(s)

    # remove duplicates and keep order
    for k in list(categorized.keys()):
        seen = set(); out = []
        for x in categorized[k]:
            if x not in seen:
                seen.add(x); out.append(x)
        categorized[k] = out

    return dict(categorized), uncategorized

# Convenience wrapper to run end-to-end for a resume: accepts extracted_skills list and full_text
def categorize_skills_for_resume(extracted_skills, full_text=None, fuzzy_cutoff=0.7):
    cat, unc = categorize_skills(extracted_skills, full_text=full_text, fuzzy_cutoff=fuzzy_cutoff)
    # ensure all categories exist even when empty
    out = {cat_name: cat.get(cat_name, []) if isinstance(cat, dict) else [] for cat_name in SKILL_TAXONOMY.keys()}
    # Update with actual categorized keys
    out.update(cat)
    out["uncategorized"] = unc
    return out

# Small helper to merge categories back into candidate record easily
def attach_categorized_skills_to_candidate(candidate_record, extracted_skills, full_text=None):
    """
    candidate_record: dict (existing resume/candidate structure)
    adds 'skills_categorized' field and returns candidate_record
    """
    categorized = categorize_skills_for_resume(extracted_skills, full_text=full_text)
    candidate_record["skills_categorized"] = categorized
    # Also add flattened technical_skills list for backwards-compatibility
    tech_keys = [k for k in SKILL_TAXONOMY.keys() if k != "soft_skills" and k != "certifications"]
    technical_flat = []
    for k in tech_keys:
        technical_flat.extend(categorized.get(k, []))
    candidate_record["technical_skills_flat"] = list(dict.fromkeys(technical_flat))  # dedupe preserving order
    candidate_record["soft_skills"] = categorized.get("soft_skills", [])
    return candidate_record
