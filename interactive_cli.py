# interactive_cli.py - Works with dynamic role matching
"""
Interactive CLI with automatic role detection (no keyword updates needed!)
"""

import sys
import time
import traceback
from typing import Optional, List, Dict, Any

try:
    from parse_query_gemma import parse_query_with_gemma_local as parse_query_with_gemma
except Exception:
    parse_query_with_gemma = None

try:
    from chatbot_poc import (
        semantic_search,
        explain_match,  # Use the explain_match function you already have
        QDRANT_COLLECTION, 
        QDRANT_HOST, 
        QDRANT_PORT
    )
    get_role_match_explanation = None  # Not available
    ROLE_TEMPLATES = {}  # Not using templates
except Exception:
    semantic_search = None
    get_role_match_explanation = None
    ROLE_TEMPLATES = {}
    QDRANT_COLLECTION = None
    QDRANT_HOST = None
    QDRANT_PORT = None

import re


def _format_candidate_dynamic(candidate: dict, rank: int, query: str) -> str:
    """Format candidate with semantic match explanations."""
    pl = candidate.get("payload") or {}
    
    name = pl.get("candidate_name") or pl.get("filename") or "Unknown Candidate"
    email = pl.get("email") or "Not provided"
    
    # Years of experience
    tys = pl.get("total_years_experience")
    if isinstance(tys, (int, float)) and 0 <= tys <= 60:
        years_str = f"{tys:.1f} years of experience"
    else:
        years_str = "Experience not specified"
    
    # Skills
    skills = pl.get("skills_set") or pl.get("skills") or []
    if isinstance(skills, dict):
        skills_flat = []
        for v in skills.values():
            if isinstance(v, (list, tuple)):
                skills_flat.extend([str(x) for x in v if x])
        skills = skills_flat
    
    relevant_skills = [str(s) for s in skills[:10] if s]
    skills_str = ", ".join(relevant_skills) if relevant_skills else "Not specified"
    
    # Match scores
    score = candidate.get("score", 0.0)
    semantic_relevance = candidate.get("semantic_relevance", score)
    
    match_quality = "Excellent" if score > 0.75 else "Strong" if score > 0.6 else "Good" if score > 0.45 else "Fair"
    
    # Build output
    output = f"""
{'='*80}
CANDIDATE #{rank}: {name}
{'='*80}

📧 Contact: {email}
⏱️  Experience: {years_str}
🎯 Match Score: {match_quality} ({score:.3f})
🔍 Semantic Relevance: {semantic_relevance:.3f}

💼 Key Skills:
   {skills_str}

✅ Why This Candidate Matches:
"""
    
    # Use the explain_match function from chatbot_poc.py
    try:
        from chatbot_poc import explain_match
        explanations = explain_match(candidate, query)
        for exp in explanations:
            output += f"   {exp}\n"
    except Exception:
        # Fallback
        output += f"   • Strong semantic similarity to your query\n"
        if relevant_sections := candidate.get("relevant_sections"):
            output += f"   • Relevant sections: {', '.join(relevant_sections)}\n"
    
    output += "\n"
    return output


def _print_results_dynamic(results: list, query: str):
    """Print results with dynamic role awareness."""
    if not results:
        print("\n❌ No matching candidates found for your query.")
        print("\n💡 Suggestions:")
        print("  • Try using more general terms")
        print("  • Remove specific experience requirements")
        print("  • Check if candidates with this profile exist in your database")
        print("  • Try adjusting role_threshold (current: 0.35)")
        return
    
    # Detect if this was a role-specific search
    query_lower = query.lower()
    role_detected = None
    for role_key in ROLE_TEMPLATES.keys():
        role_name = role_key.replace('_', ' ')
        if role_name in query_lower:
            role_detected = role_name.title()
            break
    
    role_str = f" for {role_detected}" if role_detected else ""
    
    print(f"\n✅ Found {len(results)} matching candidate(s){role_str}")
    print(f"Query: \"{query}\"")
    print(f"\nShowing top {min(len(results), 10)} results:\n")
    
    for i, c in enumerate(results[:10], start=1):
        try:
            print(_format_candidate_dynamic(c, i, query))
        except Exception as e:
            print(f"\n[Error formatting candidate {i}: {e}]\n")
    
    if len(results) > 10:
        print(f"\n... and {len(results) - 10} more candidates (use top_k to see more)")
    
    # Show statistics
    if results:
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        print(f"\n{'='*80}")
        print(f"📊 Statistics:")
        print(f"   • Total matches: {len(results)}")
        print(f"   • Average match score: {avg_score:.3f}")
        
        # Role match stats
        role_scores = [r.get("role_match_score") for r in results if r.get("role_match_score") is not None]
        if role_scores:
            avg_role_score = sum(role_scores) / len(role_scores)
            print(f"   • Average role relevance: {avg_role_score:.3f}")
        
        print(f"{'='*80}\n")


def search_from_free_text(user_query: str, 
                         top_k: int = 20,
                         role_threshold: float = 0.30,  # Maps to relevance_threshold
                         strict: bool = True):
    """
    Search with semantic matching.
    
    Args:
        user_query: Natural language query
        top_k: Number of results to return
        role_threshold: Semantic similarity threshold (0.0-1.0)
                       Lower = more lenient, Higher = more strict
        strict: (Ignored - kept for compatibility)
    """
    if semantic_search is None:
        raise RuntimeError("semantic_search not imported. Check chatbot_poc.py")
    
    # Extract basic filters from query
    years_match = re.search(r'(\d+)\s*\+?\s*years?', user_query, re.I)
    min_years = float(years_match.group(1)) if years_match else None
    
    location_match = re.search(r'\b(?:from|in|located in)\s+([A-Za-z\s]{3,30})\b', user_query, re.I)
    location = location_match.group(1).strip() if location_match else None
    
    # Call semantic search with correct parameters
    results = semantic_search(
        query=user_query,
        top_k=top_k,
        debug=False,
        filter_location=location,
        min_years_experience=min_years,
        relevance_threshold=role_threshold  # Use this instead of role_threshold
    )
    
    return results

def _main_loop():
    """Main interactive loop with dynamic role matching."""
    print("="*80)
    print("🤖 AIResumeAnalyzer - Intelligent Resume Search")
    print("   Powered by Dynamic Semantic Role Matching")
    print("="*80)
    print("\n✨ No keyword updates needed! The system understands roles automatically.\n")
    print("Example queries:")
    print("  • 'Find data engineers with 5+ years of Spark experience'")
    print("  • 'ML engineers who have deployed models to production'")
    print("  • 'Senior software engineers from California'")
    print("  • 'DevOps engineers with Kubernetes experience'")
    print("\nAdvanced options:")
    print("  • Use 'search --lenient <query>' for broader matches")
    print("  • Use 'search --strict <query>' for stricter matches")
    print("\nType 'help' for more commands.\n")
    
    # Default settings
    role_threshold = 0.35  # Default threshold
    strict_mode = True
    
    while True:
        try:
            line = input("🔍 Query >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            return
        
        if not line:
            continue
        
        # Parse command
        parts = line.split()
        cmd = parts[0].lower() if parts else ""
        
        if cmd in ("quit", "exit", "q"):
            print("\n👋 Goodbye!")
            return
        
        if cmd == "help":
            print("\n📚 Available Commands:")
            print("  search <query>        - Search resumes (default command)")
            print("  search --lenient <q>  - More lenient matching (threshold=0.25)")
            print("  search --strict <q>   - Stricter matching (threshold=0.45)")
            print("  config                - Show current configuration")
            print("  set threshold <val>   - Set role threshold (0.0-1.0)")
            print("  set strict on/off     - Enable/disable strict mode")
            print("  roles                 - List supported role types")
            print("  ingest                - Rebuild search index")
            print("  count                 - Show number of indexed resumes")
            print("  help                  - Show this help")
            print("  quit/exit/q           - Exit\n")
            continue
        
        if cmd == "roles":
            print("\n🎭 Supported Role Types (automatically detected):")
            for role_key, template in ROLE_TEMPLATES.items():
                role_name = role_key.replace('_', ' ').title()
                # Extract first line of template as description
                desc = template.strip().split('\n')[0][:80]
                print(f"  • {role_name}")
                print(f"    {desc}...")
            print("\n💡 The system uses semantic understanding, not keywords!")
            print("   You can search for any role using natural language.\n")
            continue
        
        if cmd == "config":
            print(f"\n⚙️  Current Configuration:")
            print(f"  • Role threshold: {role_threshold:.2f}")
            print(f"  • Strict mode: {'ON' if strict_mode else 'OFF'}")
            print(f"  • Embedding model: {'Initialized' if semantic_search else 'Not available'}")
            print(f"\n💡 Lower threshold = more results, Higher threshold = more precise\n")
            continue
        
        if cmd == "set":
            if len(parts) < 3:
                print("❌ Usage: set threshold <value> OR set strict on/off\n")
                continue
            
            setting = parts[1].lower()
            if setting == "threshold":
                try:
                    new_val = float(parts[2])
                    if 0.0 <= new_val <= 1.0:
                        role_threshold = new_val
                        print(f"✓ Role threshold set to {role_threshold:.2f}\n")
                    else:
                        print("❌ Threshold must be between 0.0 and 1.0\n")
                except ValueError:
                    print("❌ Invalid threshold value\n")
            elif setting == "strict":
                if parts[2].lower() in ("on", "true", "1"):
                    strict_mode = True
                    print("✓ Strict mode enabled\n")
                elif parts[2].lower() in ("off", "false", "0"):
                    strict_mode = False
                    print("✓ Strict mode disabled\n")
                else:
                    print("❌ Use 'on' or 'off'\n")
            else:
                print("❌ Unknown setting. Use 'threshold' or 'strict'\n")
            continue
        
        if cmd == "count":
            try:
                from qdrant_client import QdrantClient
                from chatbot_poc import QDRANT_HOST as host, QDRANT_PORT as port, QDRANT_COLLECTION as coll
                client = QdrantClient(host=host, port=port)
                c = client.count(collection_name=coll, exact=True)
                total = getattr(c, "count", c if isinstance(c, int) else None)
                print(f"\n📊 Total indexed resume chunks: {total}")
                print(f"💡 Each resume is split into multiple chunks for better matching\n")
            except Exception as e:
                print(f"\n❌ Count failed: {e}\n")
            continue
        
        if cmd == "ingest":
            print("\n⚠️  Running ingestion (this will recreate the collection)...")
            print("⏳ This may take a few minutes...\n")
            try:
                import ingest_resume as ingest_mod
                ingest_mod.main()
                print("\n✅ Ingestion complete! Search index rebuilt.\n")
            except Exception as e:
                print(f"\n❌ Ingestion failed: {e}")
                traceback.print_exc()
                print()
            continue
        
        # Parse search flags
        user_query = line
        temp_threshold = role_threshold
        temp_strict = strict_mode
        
        if cmd == "search":
            if len(parts) < 2:
                print("\n❌ Please provide a search query.\n")
                continue
            
            # Check for flags
            if parts[1].startswith("--"):
                flag = parts[1][2:].lower()
                if flag == "lenient":
                    temp_threshold = 0.25
                    temp_strict = True
                    user_query = " ".join(parts[2:])
                    print(f"🔓 Using lenient mode (threshold={temp_threshold})")
                elif flag == "strict":
                    temp_threshold = 0.45
                    temp_strict = True
                    user_query = " ".join(parts[2:])
                    print(f"🔒 Using strict mode (threshold={temp_threshold})")
                else:
                    user_query = " ".join(parts[1:])
            else:
                user_query = " ".join(parts[1:])
        else:
            # Treat entire line as search query (default command)
            user_query = line
        
        if not user_query or not user_query.strip():
            print("\n❌ Empty query.\n")
            continue
        
        print(f"\n🔄 Searching: \"{user_query}\"")
        print(f"⚙️  Settings: threshold={temp_threshold:.2f}, strict={'ON' if temp_strict else 'OFF'}")
        
        t0 = time.time()
        
        try:
            results = search_from_free_text(
                user_query, 
                top_k=20,
                role_threshold=temp_threshold,
                strict=temp_strict
            )
            t1 = time.time()
            print(f"⏱️  Search completed in {t1 - t0:.2f}s\n")
            _print_results_dynamic(results, user_query)
        except Exception as e:
            print(f"\n❌ Search failed: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    _main_loop()